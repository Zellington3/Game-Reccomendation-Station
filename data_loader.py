import pandas as pd
from neo4j import GraphDatabase, basic_auth
import logging
import config # Assumes config.py has NEO4J_URI, NEO4J_USER etc.
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_neo4j_password(password_file):
    """Reads the Neo4j password from a file."""
    try:
        with open(password_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"❌ Password file '{password_file}' not found.")
        return None
    except Exception as e:
        logging.error(f"❌ Error reading password file: {e}")
        return None

def load_neo4j_data(uri=config.NEO4J_URI, user=config.NEO4J_USER,
                     password_file=config.DB_PASSWORD_FILE, min_playtime=0):
    """Loads interaction and genre dataframes from Neo4j for training."""
    logging.info("📥 Loading data from Neo4j for training pipeline...")
    password = get_neo4j_password(password_file)
    if not password:
        logging.error("❌ Neo4j password could not be read.")
        return pd.DataFrame(), pd.DataFrame()

    likes_df = pd.DataFrame()
    genre_df = pd.DataFrame()
    driver = None

    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        driver.verify_connectivity()
        logging.info("✅ Neo4j connection successful.")

        with driver.session(database="neo4j") as session:
            # Query 1: Get Likes/Interactions
            # Fetches user_id (int), game_id (str), playtime (float)
            likes_query = f"""
            MATCH (u:User)-[l:LIKES]->(g:Game)
            WHERE l.playtime IS NOT NULL AND l.playtime >= $min_playtime
            RETURN
                u.user_id AS user_id,         // Neo4j integer -> keep as int
                g.app_id AS game_id,           // Neo4j string -> keep as str
                toFloat(l.playtime) AS playtime // Neo4j int/float -> ensure float
            // ORDER BY u.user_id, g.app_id // Optional ordering
            """
            logging.info(f" Executing likes query with min_playtime={min_playtime}...")
            result_likes = session.run(likes_query, min_playtime=float(min_playtime))
            # Use list comprehension for potentially better performance
            likes_data = [record.data() for record in result_likes]
            # likes_data = [{'user_id': r['user_id'], 'game_id': r['game_id'], 'playtime': r['playtime']}
            #               for r in result_likes]
            if likes_data:
                 likes_df = pd.DataFrame(likes_data)
                 # Explicitly set types after DataFrame creation for safety
                 likes_df['user_id'] = likes_df['user_id'].astype(int)
                 likes_df['game_id'] = likes_df['game_id'].astype(str)
                 likes_df['playtime'] = likes_df['playtime'].astype(float)
                 logging.info(f"📊 Loaded {len(likes_df)} 'likes' interactions (playtime >= {min_playtime}).")
            else:
                 logging.warning("⚠️ No 'likes' interactions found matching the criteria.")
                 likes_df = pd.DataFrame(columns=['user_id', 'game_id', 'playtime']) # Ensure empty df has columns


            # Query 2: Get Genre Relationships
            # Fetches game_id (str), genre (str)
            genre_query = """
            MATCH (g:Game)-[:HAS_GENRE]->(ge:Genre)
            // WHERE g.app_id IS NOT NULL AND ge.name IS NOT NULL // Constraints should ensure this
            RETURN
                g.app_id AS game_id, // Neo4j string -> keep as str
                ge.name AS genre    // Neo4j string -> keep as str
            // ORDER BY g.app_id, ge.name // Optional ordering
            """
            logging.info(" Executing genre query...")
            result_genre = session.run(genre_query)
            genre_data = [record.data() for record in result_genre]
            # genre_data = [{'game_id': r['game_id'], 'genre': r['genre']} for r in result_genre]
            if genre_data:
                 genre_df = pd.DataFrame(genre_data)
                 # Ensure types
                 genre_df['game_id'] = genre_df['game_id'].astype(str)
                 genre_df['genre'] = genre_df['genre'].astype(str)
                 logging.info(f"📊 Loaded {len(genre_df)} game-genre relations.")
            else:
                 logging.warning("⚠️ No game-genre relationships found.")
                 genre_df = pd.DataFrame(columns=['game_id', 'genre']) # Ensure empty df has columns


    except neo4j.exceptions.AuthError as e:
        logging.error(f"❌ Neo4j authentication failed: {e}")
    except neo4j.exceptions.ServiceUnavailable as e:
        logging.error(f"❌ Neo4j connection failed: {e}")
    except Exception as e:
        logging.error(f"❌ Error during Neo4j data loading: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logging.info("🚪 Neo4j connection closed.")

    # Final Validation 
    if not likes_df.empty:
        if likes_df['user_id'].isnull().any() or likes_df['game_id'].isnull().any():
            logging.warning("⚠️ Likes data contains unexpected null user_ids or game_ids after load. Check query/data.")
    if not genre_df.empty:
         if genre_df['game_id'].isnull().any() or genre_df['genre'].isnull().any():
             logging.warning("⚠️ Genre data contains unexpected null game_ids or genres after load. Check query/data.")

    return likes_df, genre_df
