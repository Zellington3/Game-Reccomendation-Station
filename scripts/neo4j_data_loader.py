import pandas as pd
from neo4j import GraphDatabase, basic_auth
import logging
import os

# This assumes config.py is in the same directory or Python path
# and contains NEO4J_URI, NEO4J_USER, DB_PASSWORD_FILE
try:
    import config
except ImportError:
    logging.error("❌ config.py not found. Please ensure it exists and contains necessary settings.")
    # Provide safe defaults or exit if config is critical
    class ConfigMock:
        NEO4J_URI="bolt://localhost:7687"
        NEO4J_USER="neo4j"
        DB_PASSWORD_FILE="dbpassword.txt"
        MIN_PLAYTIME = 0 # Add default for min_playtime if used directly
    config = ConfigMock()
    logging.warning("Using default config values as config.py was not found.")

# Logging is configured by the main script (train.py)
def get_neo4j_password(password_file=config.DB_PASSWORD_FILE):
    """Reads the Neo4j password from the specified file."""
    password = None
    # Assumes password file is in the same dir as the script being run or configured path is relative/absolute
    file_path_to_check = password_file
    logging.debug(f"Attempting to read password from file: {os.path.abspath(file_path_to_check)}")
    try:
        with open(file_path_to_check, "r") as f:
            password = f.read().strip()
        if not password:
             logging.error(f"❌ Password file '{file_path_to_check}' is empty.")
             return None
        logging.debug(f"Password successfully read from file '{file_path_to_check}'.")
        return password
    except FileNotFoundError:
        logging.error(f"❌ Password file '{os.path.abspath(file_path_to_check)}' not found.")
        logging.error(f"  Ensure '{password_file}' exists relative to the script execution directory or provide an absolute path in config.py.")
        return None
    except Exception as e:
        logging.error(f"❌ Error reading password file '{password_file}': {e}")
        return None

def load_neo4j_data(uri=config.NEO4J_URI, user=config.NEO4J_USER,
                     password_file=config.DB_PASSWORD_FILE, min_playtime=config.MIN_PLAYTIME):
    """Loads interaction and genre dataframes from Neo4j for training."""
    # These logging calls will use the configuration set by train.py
    logging.info("📥 Loading data from Neo4j for training pipeline...")
    password = get_neo4j_password(password_file) # Uses the function argument, defaulting to config.DB_PASSWORD_FILE
    if not password:
        logging.error("❌ Neo4j password could not be obtained.")
        return pd.DataFrame(columns=['user_id', 'game_id', 'playtime']), pd.DataFrame(columns=['game_id', 'genre']) # Return empty with columns

    likes_df = pd.DataFrame()
    genre_df = pd.DataFrame()
    driver = None

    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password), connection_timeout=30.0)
        driver.verify_connectivity()
        logging.info("✅ Neo4j connection successful.")

        with driver.session(database="neo4j") as session:
            # Query 1: Get Likes/Interactions
            likes_query = f"""
            MATCH (u:User)-[l:LIKES]->(g:Game)
            WHERE l.playtime IS NOT NULL AND l.playtime >= $min_playtime
            RETURN
                u.user_id AS user_id,
                g.app_id AS game_id,
                toFloat(l.playtime) AS playtime
            """
            logging.info(f" Executing likes query with min_playtime={min_playtime}...")
            result_likes = session.run(likes_query, min_playtime=float(min_playtime))
            likes_data = [record.data() for record in result_likes]
            if likes_data:
                 likes_df = pd.DataFrame(likes_data)
                 likes_df['user_id'] = likes_df['user_id'].astype(int)
                 likes_df['game_id'] = likes_df['game_id'].astype(str)
                 likes_df['playtime'] = likes_df['playtime'].astype(float)
                 logging.info(f"📊 Loaded {len(likes_df)} 'likes' interactions (playtime >= {min_playtime}).")
            else:
                 logging.warning("⚠️ No 'likes' interactions found matching the criteria.")
                 likes_df = pd.DataFrame(columns=['user_id', 'game_id', 'playtime'])

            # Query 2: Get Genre Relationships
            genre_query = """
            MATCH (g:Game)-[:HAS_GENRE]->(ge:Genre)
            RETURN g.app_id AS game_id, ge.name AS genre
            """
            logging.info(" Executing genre query...")
            result_genre = session.run(genre_query)
            genre_data = [record.data() for record in result_genre]
            if genre_data:
                 genre_df = pd.DataFrame(genre_data)
                 genre_df['game_id'] = genre_df['game_id'].astype(str)
                 genre_df['genre'] = genre_df['genre'].astype(str)
                 logging.info(f"📊 Loaded {len(genre_df)} game-genre relations.")
            else:
                 logging.warning("⚠️ No game-genre relationships found.")
                 genre_df = pd.DataFrame(columns=['game_id', 'genre'])

    except neo4j.exceptions.AuthError as e: logging.error(f"❌ Neo4j authentication failed for user '{user}'. Check credentials ({password_file}). Details: {e}")
    except neo4j.exceptions.ServiceUnavailable as e: logging.error(f"❌ Could not connect to Neo4j at {uri}. Details: {e}")
    except Exception as e: logging.error(f"❌ Error during Neo4j data loading: {e}", exc_info=True)
    finally:
        if driver: driver.close(); logging.info("🚪 Neo4j connection closed.")

    # Return empty dataframes with defined columns if loading failed
    if likes_df.empty: likes_df = pd.DataFrame(columns=['user_id', 'game_id', 'playtime'])
    if genre_df.empty: genre_df = pd.DataFrame(columns=['game_id', 'genre'])

    # Check for nulls after potential partial load
    if not likes_df.empty and (likes_df['user_id'].isnull().any() or likes_df['game_id'].isnull().any()):
        logging.warning("⚠️ Likes data contains null user_ids or game_ids after load. Check query/data.")
    if not genre_df.empty and (genre_df['game_id'].isnull().any() or genre_df['genre'].isnull().any()):
        logging.warning("⚠️ Genre data contains null game_ids or genres after load. Check query/data.")

    return likes_df, genre_df
