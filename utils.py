# utils.py
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pickle
import os
import logging
import time
import neo4j 

import config

# Ensure logging is configured (can be set here or in train.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sparse_matrices(likes_df, test_size=0.2, random_state=42):
    """Splits data and creates training and test sparse matrices using log-transformed playtime."""
    if likes_df.empty:
        logging.error("❌ Cannot create matrices from empty DataFrame.")
        return None, None, None, None, None

    logging.info(f"Preparing sparse matrices from {len(likes_df)} interactions...")

    # Ensure consistent categories across train and test
    all_users = sorted(likes_df['user_id'].unique())
    all_games = sorted(likes_df['game_id'].unique())
    user_cat_type = pd.CategoricalDtype(categories=all_users, ordered=True)
    game_cat_type = pd.CategoricalDtype(categories=all_games, ordered=True)

    # Map original IDs to internal integer indices (0 to N-1)
    user_map = {user: i for i, user in enumerate(all_users)}
    game_map = {game: i for i, game in enumerate(all_games)}
    # Inverse mapping needed for recommendations later
    game_id_map = {i: game for game, i in game_map.items()}

    logging.info(f" Mapped {len(user_map)} unique users and {len(game_map)} unique games.")

    likes_df['user_cat'] = likes_df['user_id'].astype(user_cat_type).cat.codes
    likes_df['game_cat'] = likes_df['game_id'].astype(game_cat_type).cat.codes

    # Apply log transformation to playtime to act as confidence score C = 1 + alpha * log(1 + playtime)
    # Using alpha=1 here for simplicity, effectively C = 1 + log(1+playtime)
    likes_df['confidence'] = 1.0 + np.log1p(likes_df['playtime'].astype(np.float32))

    logging.info(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    # Split data *after* creating categorical codes and confidence
    try:
        # Stratify only if each user has at least 2 interactions, otherwise split fails
        user_counts = likes_df['user_id'].value_counts()
        stratify_col = likes_df['user_id'] if all(user_counts >= 2) else None

        train_df, test_df = train_test_split(
            likes_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        if stratify_col is None and test_size > 0:
             logging.warning("⚠️ Could not stratify split (some users might have < 2 interactions). Performing non-stratified split.")

    except ValueError as e: # Catch potential errors during split
        logging.error(f"❌ Error during train/test split: {e}. Check data consistency.")
        return None, None, None, None, None

    logging.info(f" Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    # Create sparse matrices
    shape = (len(user_map), len(game_map)) # users x items

    train_matrix = sparse.csr_matrix(
        (train_df["confidence"].astype(np.float32),
         (train_df["user_cat"], train_df["game_cat"])),
        shape=shape
    )

    # Test matrix: For evaluation, use 1 for observed interactions, 0 otherwise
    test_matrix = sparse.csr_matrix(
        (np.ones(len(test_df), dtype=np.float32), # Use 1 for relevance check
         (test_df["user_cat"], test_df["game_cat"])),
        shape=shape
    )

    logging.info(f" Created Train Matrix (Users x Items): {train_matrix.shape}, nnz={train_matrix.nnz}")
    logging.info(f" Created Test Matrix (Users x Items): {test_matrix.shape}, nnz={test_matrix.nnz}")

    # Save test data interactions if needed for other analysis
    test_interactions_path = os.path.join(config.MODEL_DIR, "test_interactions.csv")
    try:
        test_df[['user_id', 'game_id', 'playtime']].to_csv(test_interactions_path, index=False)
        logging.info(f" Test interactions saved to {test_interactions_path}")
    except Exception as e:
        logging.warning(f" Could not save test interactions to CSV: {e}")


    return train_matrix, test_matrix, user_map, game_map, game_id_map

def build_genre_similarity_matrix(genre_df, game_map):
    """Builds the genre similarity matrix, aligning with game_map."""
    if genre_df.empty or not game_map:
        logging.warning("⚠️ Genre DataFrame is empty or game_map is missing. Skipping genre matrix build.")
        return None, None

    logging.info("🎨 Building genre similarity matrix...")

    known_games = set(game_map.keys())
    genre_df_filtered = genre_df[genre_df['game_id'].isin(known_games)].copy()

    if genre_df_filtered.empty:
        logging.warning("⚠️ No genre information found for the games present in the interaction data (game_map). Skipping genre matrix.")
        return None, None

    # Create categories based *only* on the filtered games with genre info
    genre_games = sorted(genre_df_filtered['game_id'].unique())
    all_genres = sorted(genre_df_filtered['genre'].unique())
    genre_game_cat_type = pd.CategoricalDtype(categories=genre_games, ordered=True)
    genre_cat_type = pd.CategoricalDtype(categories=all_genres, ordered=True)

    genre_df_filtered['game_cat'] = genre_df_filtered['game_id'].astype(genre_game_cat_type).cat.codes
    genre_df_filtered['genre_cat'] = genre_df_filtered['genre'].astype(genre_cat_type).cat.codes

    # Lookup for *this specific matrix* (mapping game_id to its index in the genre matrix)
    genre_game_lookup = {game: i for i, game in enumerate(genre_game_cat_type.categories)}

    shape = (len(genre_game_lookup), len(genre_cat_type.categories)) # games_with_genres x genres

    game_genre_matrix = sparse.csr_matrix(
        (np.ones(len(genre_df_filtered)),
         (genre_df_filtered['game_cat'], genre_df_filtered['genre_cat'])),
        shape=shape,
        dtype=np.float32
    )

    # Calculate Cosine Similarity: (X * X^T) after normalization
    logging.info(" Normalizing genre matrix and calculating similarity...")
    normalized_matrix = normalize(game_genre_matrix, norm='l2', axis=1)
    similarity_matrix = normalized_matrix.dot(normalized_matrix.T).tocsr()
    logging.info(f"✅ Genre similarity matrix built. Shape: {similarity_matrix.shape} for {len(genre_game_lookup)} games.")

    return similarity_matrix, genre_game_lookup

def save_model_artifacts(model_dir, model, user_map, game_map, game_id_map, genre_sim_matrix, genre_game_lookup, metrics=None, best_params=None):
    """Saves all necessary components including the model object and metadata."""
    logging.info(f"💾 Saving model artifacts to {model_dir}...")
    os.makedirs(model_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        # --- Save the complete model object ---
        model_path = os.path.join(model_dir, "als_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f" Saved complete ALS model object to {model_path}")
        # ------------------------------------

        # Save factors (optional, can be extracted from model object, but useful)
        np.save(os.path.join(model_dir, "item_factors.npy"), model.item_factors)
        np.save(os.path.join(model_dir, "user_factors.npy"), model.user_factors)

        # Save mappings
        with open(os.path.join(model_dir, "user_map.pkl"), "wb") as f:
            pickle.dump(user_map, f) # Map UserID -> internal index (for reference)
        with open(os.path.join(model_dir, "game_map.pkl"), "wb") as f:
            pickle.dump(game_map, f) # Map GameID -> internal index (needed for recommend)
        with open(os.path.join(model_dir, "game_id_map.pkl"), "wb") as f:
            pickle.dump(game_id_map, f) # Map internal index -> GameID (needed for recommend)

        # Save genre similarity if it exists
        if genre_sim_matrix is not None and genre_game_lookup is not None:
             sparse.save_npz(os.path.join(model_dir, "genre_sim_matrix.npz"), genre_sim_matrix)
             with open(os.path.join(model_dir, "genre_game_lookup.pkl"), "wb") as f:
                pickle.dump(genre_game_lookup, f)
        else:
            logging.info("⚠️ Genre similarity matrix or lookup is None, skipping save.") # Use info level

        # --- Save Metadata (Metrics & Parameters) ---
        metadata = {
            "timestamp": timestamp,
            "metrics": metrics or {},
            "best_params": best_params or {},
            "evaluation_k": config.EVALUATION_K,
            "min_playtime": config.MIN_PLAYTIME,
            "random_seed": config.RANDOM_SEED,
            "neo4j_uri": config.NEO4J_URI, # Record data source
            # Add implicit version? sys.modules['implicit'].__version__
        }
        metadata_path = os.path.join(model_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
             pickle.dump(metadata, f)
        logging.info(f" Saved metadata (metrics, params) to {metadata_path}")
        # Also save metrics to a human-readable text file
        if metrics:
            metrics_txt_path = os.path.join(model_dir, "evaluation_metrics.txt")
            with open(metrics_txt_path, "w") as f:
                 f.write(f"Timestamp: {timestamp}\n")
                 f.write(f"Best Params: {best_params}\n")
                 f.write("Metrics:\n")
                 for name, value in metrics.items():
                      f.write(f"  {name}: {value:.4f}\n")
            logging.info(f" Saved human-readable metrics to {metrics_txt_path}")
        # --- ---

        logging.info(f"✅ Model artifacts successfully saved to ./{model_dir}")
        return True
    except pickle.PicklingError as e:
         logging.error(f"❌ Failed to pickle model object: {e}. Check if model is picklable.")
         return False
    except Exception as e:
        logging.error(f"❌ Failed to save model artifacts: {type(e).__name__} - {e}")
        return False

def load_model_artifacts(model_dir):
    """Loads artifacts including the model object and metadata."""
    logging.info(f"📦 Loading model artifacts from {model_dir}...")
    artifacts = {}
    required_files = {
        'model': os.path.join(model_dir, "als_model.pkl"),
        'game_map': os.path.join(model_dir, "game_map.pkl"),
        'game_id_map': os.path.join(model_dir, "game_id_map.pkl"),
    }
    optional_files = {
         'user_map': os.path.join(model_dir, "user_map.pkl"),
         'genre_sim_matrix': os.path.join(model_dir, "genre_sim_matrix.npz"),
         'genre_game_lookup': os.path.join(model_dir, "genre_game_lookup.pkl"),
         'metadata': os.path.join(model_dir, "metadata.pkl"),
         'item_factors': os.path.join(model_dir, "item_factors.npy"), # Redundant but can load
         'user_factors': os.path.join(model_dir, "user_factors.npy"), # Redundant
    }

    try:
        # Load required files
        for key, path in required_files.items():
            if not os.path.exists(path):
                 logging.error(f"❌ Required artifact file not found: {path}")
                 raise FileNotFoundError(f"Missing required artifact: {path}")
            if path.endswith(".pkl"):
                with open(path, "rb") as f:
                    artifacts[key] = pickle.load(f)
            # Add handlers for other types if needed later
            logging.debug(f" Loaded required artifact: {key} from {path}")

        # Load optional files
        for key, path in optional_files.items():
            if os.path.exists(path):
                try:
                    if path.endswith(".pkl"):
                        with open(path, "rb") as f:
                            artifacts[key] = pickle.load(f)
                    elif path.endswith(".npz"):
                         artifacts[key] = sparse.load_npz(path)
                    elif path.endswith(".npy"):
                         artifacts[key] = np.load(path)
                    logging.debug(f" Loaded optional artifact: {key} from {path}")
                except Exception as e:
                     logging.warning(f"⚠️ Could not load optional artifact {key} from {path}: {e}")
                     artifacts[key] = None
            else:
                logging.info(f" Optional artifact file not found, skipping: {path}")
                artifacts[key] = None

        # Extract factors from loaded model for easier access if needed
        if 'model' in artifacts and artifacts['model'] is not None:
             if 'item_factors' not in artifacts or artifacts['item_factors'] is None:
                  artifacts['item_factors'] = artifacts['model'].item_factors
             if 'user_factors' not in artifacts or artifacts['user_factors'] is None:
                  artifacts['user_factors'] = artifacts['model'].user_factors

        logging.info("✅ Model artifacts loaded successfully.")
        return artifacts

    except FileNotFoundError as e:
        # Already logged the specific missing file
        return None
    except (pickle.UnpicklingError, EOFError) as e: # Catch EOFError too
         logging.error(f"❌ Failed to unpickle artifact from {path}: {e}. File might be corrupt or incompatible.")
         return None
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during artifact loading: {type(e).__name__} - {e}")
        return None