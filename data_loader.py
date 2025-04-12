# data_loader.py
import pandas as pd
from neo4j import GraphDatabase, basic_auth
import logging
import config
import os 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_neo4j_data(uri, user, password_file, min_playtime=0):
    """Loads interaction and genre data from Neo4j."""
    logging.info("📥 Loading data from Neo4j...")
    password = None
    try:
        with open(password_file, "r") as f:
            password = f.read().strip()
    except FileNotFoundError:
        logging.error(f"❌ Password file '{password_file}' not found.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logging.error(f"❌ Error reading password file: {e}")
        return pd.DataFrame(), pd.DataFrame()

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
            likes_query = f"""
            MATCH (u:User)-[l:LIKES]->(g:Game)
            WHERE l.playtime IS NOT NULL AND l.playtime >= $min_playtime
            RETURN
                toInteger(u.user_id) AS user_id,
                toString(g.app_id) AS game_id,
                toFloat(l.playtime) AS playtime
            """
            logging.info(f"Executing likes query with min_playtime={min_playtime}...")
            likes_result = session.run(likes_query, min_playtime=float(min_playtime))
            likes_data = [{'user_id': r['user_id'], 'game_id': r['game_id'], 'playtime': r['playtime']}
                          for r in likes_result]
            likes_df = pd.DataFrame(likes_data)
            logging.info(f"📊 Loaded {len(likes_df)} 'likes' interactions (playtime >= {min_playtime}).")

            # Fetch Genre data
            genre_query = """
            MATCH (g:Game)-[:HAS_GENRE]->(ge:Genre)
            WHERE g.app_id IS NOT NULL AND ge.name IS NOT NULL
            RETURN toString(g.app_id) AS game_id, ge.name AS genre
            """
            logging.info("Executing genre query...")
            genre_result = session.run(genre_query)
            genre_data = [{'game_id': r['game_id'], 'genre': r['genre']} for r in genre_result]
            genre_df = pd.DataFrame(genre_data)
            logging.info(f"📊 Loaded {len(genre_df)} genre relations.")

    # Catches specific Neo4j exceptions
    except neo4j.exceptions.AuthError as e:
        logging.error(f"❌ Neo4j authentication failed for user '{user}'. Check credentials ({password_file}). Details: {e}")
    except neo4j.exceptions.ServiceUnavailable as e:
        logging.error(f"❌ Could not connect to Neo4j at {uri}. Ensure the server is running and accessible. Details: {e}")
    except Exception as e:
        # Catches other potential errors during query execution or data handling
        logging.error(f"❌ An unexpected error occurred during Neo4j interaction: {type(e).__name__} - {e}")
    finally:
        if driver:
            driver.close()
            logging.info("🚪 Neo4j connection closed.")

    # Data Validation
    if not likes_df.empty:
        if likes_df['user_id'].isnull().any() or likes_df['game_id'].isnull().any():
            logging.warning("⚠️ Likes data contains null user_ids or game_ids after load. Cleaning...")
            initial_rows = len(likes_df)
            likes_df.dropna(subset=['user_id', 'game_id'], inplace=True)
            logging.info(f" Removed {initial_rows - len(likes_df)} rows with null IDs.")
        if likes_df['playtime'].isnull().any():
             null_playtime_count = likes_df['playtime'].isnull().sum()
             logging.warning(f"⚠️ Found {null_playtime_count} null playtimes. Imputing with minimum configured ({min_playtime}).")
             likes_df['playtime'].fillna(min_playtime, inplace=True) # Or consider removing these rows
    else:
        # Only log warning if no error occurred during loading
        if 'driver' in locals() and driver is not None: # Check if connection attempt was made
             logging.warning("⚠️ No 'likes' data loaded from Neo4j (or all filtered out by min_playtime).")

    if not genre_df.empty:
         if genre_df['game_id'].isnull().any() or genre_df['genre'].isnull().any():
             logging.warning("⚠️ Genre data contains null game_ids or genres after load. Cleaning...")
             initial_rows = len(genre_df)
             genre_df.dropna(subset=['game_id', 'genre'], inplace=True)
             logging.info(f" Removed {initial_rows - len(genre_df)} rows with null genre info.")
    else:
        if 'driver' in locals() and driver is not None:
             logging.warning("⚠️ No genre data loaded from Neo4j.")

    return likes_df, genre_df