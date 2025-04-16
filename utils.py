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

# Assuming config.py exists and has necessary variables like MODEL_DIR etc.
try:
    import config
except ImportError:
    # Provide defaults if config is missing, or raise error
    logging.error("config.py not found or has errors. Using defaults.")
    class ConfigMock:
        MODEL_DIR = "model"
        EVALUATION_K = 10
        MIN_PLAYTIME = 5
        RANDOM_SEED = 42
        NEO4J_URI = "bolt://localhost:7687"
        # Add other necessary defaults from config
    config = ConfigMock()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- create_sparse_matrices (Unchanged from previous correct version) ---
def create_sparse_matrices(likes_df, test_size=0.2, random_state=42,
                           create_full_split=False,
                           existing_user_map=None,
                           existing_game_map=None):
    if likes_df is None or likes_df.empty:
        logging.error("❌ Cannot create matrices from empty or None DataFrame.")
        return None
    logging.info(f"Preparing sparse matrices from {len(likes_df)} interactions...")
    try:
        likes_df['user_id'] = likes_df['user_id'].astype(int)
        likes_df['game_id'] = likes_df['game_id'].astype(str)
    except Exception as e:
        logging.error(f"❌ Error converting ID columns to expected types: {e}"); return None

    if existing_user_map is not None and existing_game_map is not None:
        logging.info("Using provided user and game maps.")
        user_map = existing_user_map; game_map = existing_game_map
        game_id_map = {i: game for game, i in game_map.items()}
        known_users = set(user_map.keys()); known_games = set(game_map.keys())
        initial_rows = len(likes_df)
        likes_df = likes_df[likes_df['user_id'].isin(known_users) & likes_df['game_id'].isin(known_games)].copy()
        rows_filtered = initial_rows - len(likes_df)
        if rows_filtered > 0: logging.warning(f" Filtered out {rows_filtered} interactions (IDs not in provided maps).")
        if likes_df.empty: logging.error("❌ Input DataFrame empty after filtering against provided maps."); return None
    else:
        logging.info("Creating new user and game maps from the input DataFrame.")
        all_users = sorted(likes_df['user_id'].unique()); all_games = sorted(likes_df['game_id'].unique())
        if not all_users or not all_games: logging.error("❌ No users or games found in input DataFrame."); return None
        user_map = {user: i for i, user in enumerate(all_users)}; game_map = {game: i for i, game in enumerate(all_games)}
        game_id_map = {i: game for game, i in game_map.items()}

    num_users = len(user_map); num_games = len(game_map)
    logging.info(f" Using maps with {num_users} unique users and {num_games} unique games.")

    likes_df['user_cat'] = likes_df['user_id'].map(user_map)
    likes_df['game_cat'] = likes_df['game_id'].map(game_map)
    rows_before_dropna = len(likes_df)
    likes_df.dropna(subset=['user_cat', 'game_cat'], inplace=True)
    if rows_before_dropna > len(likes_df): logging.warning(f" Dropped {rows_before_dropna - len(likes_df)} rows due to missing mapping.")
    if likes_df.empty: logging.error("❌ DataFrame empty after mapping."); return None
    likes_df['user_cat'] = likes_df['user_cat'].astype(int); likes_df['game_cat'] = likes_df['game_cat'].astype(int)
    likes_df['confidence'] = 1.0 + np.log1p(likes_df['playtime'].astype(np.float32))

    logging.info(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    train_df, test_df = None, None
    try:
        user_counts = likes_df['user_id'].value_counts()
        can_stratify = test_size > 0 and not user_counts.empty and all(user_counts >= 2)
        stratify_col = likes_df['user_id'] if can_stratify else None
        if len(likes_df) > 1 and test_size < 1.0 and test_size > 0.0:
             train_df, test_df = train_test_split(likes_df, test_size=test_size, random_state=random_state, stratify=stratify_col)
        elif test_size == 0.0: train_df = likes_df; test_df = pd.DataFrame(columns=likes_df.columns)
        elif test_size >= 1.0: train_df = pd.DataFrame(columns=likes_df.columns); test_df = likes_df
        else: train_df = likes_df; test_df = pd.DataFrame(columns=likes_df.columns) # Only one row
        if stratify_col is None and test_size > 0 and len(likes_df) > 1: logging.warning("⚠️ Could not stratify split.")
        logging.info(f" Data split sizes - Train: {len(train_df)}, Test: {len(test_df)}")
    except Exception as e: logging.error(f"❌ Error during split: {e}", exc_info=True); return None

    shape = (num_users, num_games)
    train_matrix = sparse.csr_matrix(shape, dtype=np.float32); test_matrix = sparse.csr_matrix(shape, dtype=np.float32)
    if not train_df.empty: train_matrix = sparse.csr_matrix((train_df["confidence"].astype(np.float32), (train_df["user_cat"], train_df["game_cat"])), shape=shape)
    if not test_df.empty: test_matrix = sparse.csr_matrix((np.ones(len(test_df), dtype=np.float32), (test_df["user_cat"], test_df["game_cat"])), shape=shape)

    if create_full_split:
        logging.info(f" Created Full Train Matrix: {train_matrix.shape}, nnz={train_matrix.nnz}")
        logging.info(f" Created Full Test Matrix: {test_matrix.shape}, nnz={test_matrix.nnz}")
        return train_matrix, test_matrix, user_map, game_map, game_id_map, train_df, test_df
    else:
        logging.info(f" Created Train Matrix (for HPO/Sample): {train_matrix.shape}, nnz={train_matrix.nnz}")
        logging.info(f" Created Test Matrix (for HPO/Sample): {test_matrix.shape}, nnz={test_matrix.nnz}")
        # Optional save HPO test interactions
        test_interactions_path = os.path.join(config.MODEL_DIR, "hpo_test_interactions.csv")
        try:
             if not test_df.empty:
                  test_df[['user_id', 'game_id', 'playtime']].to_csv(test_interactions_path, index=False)
                  logging.info(f" HPO test interactions saved to {test_interactions_path}")
        except Exception as e: logging.warning(f" Could not save HPO test interactions: {e}")
        return train_matrix, test_matrix, user_map, game_map, game_id_map


# --- build_genre_similarity_matrix (Unchanged from previous correct version) ---
def build_genre_similarity_matrix(genre_df, game_map):
    if genre_df is None or genre_df.empty: logging.warning("⚠️ Genre DataFrame empty. Skipping genre matrix."); return None, None
    if not game_map: logging.warning("⚠️ game_map missing. Skipping genre matrix."); return None, None
    logging.info("🎨 Building genre similarity matrix...")
    try:
        if 'game_id' not in genre_df.columns or 'genre' not in genre_df.columns: raise ValueError("Genre DF needs 'game_id', 'genre'")
        genre_df['game_id'] = genre_df['game_id'].astype(str); genre_df = genre_df.dropna(subset=['genre']); genre_df['genre'] = genre_df['genre'].astype(str)
    except Exception as e: logging.error(f"❌ Error preparing genre_df columns: {e}"); return None, None

    known_games = set(game_map.keys())
    logging.info(f" Filtering genre_df against {len(known_games)} known game IDs from interactions map.")
    genre_df_filtered = genre_df[genre_df['game_id'].isin(known_games)].copy()
    if genre_df_filtered.empty: logging.warning("⚠️ No genre info found for games in game_map AFTER filtering."); return None, None
    logging.info(f" Found {len(genre_df_filtered)} genre rows corresponding to known games.")
    logging.info(f" Number of unique games with genres found: {genre_df_filtered['game_id'].nunique()}")

    # Optional: Handle multi-genre strings (e.g., explode ';') - Adjust separator if needed
    # if genre_df_filtered['genre'].str.contains(';', na=False).any():
    #      logging.info(" Splitting multi-genre strings on ';'")
    #      genre_df_filtered = genre_df_filtered.assign(genre=genre_df_filtered['genre'].str.split(';')).explode('genre')
    #      genre_df_filtered['genre'] = genre_df_filtered['genre'].str.strip()
    #      genre_df_filtered = genre_df_filtered[genre_df_filtered['genre'] != '']

    genre_games_present = sorted(genre_df_filtered['game_id'].unique())
    all_genres_present = sorted(genre_df_filtered['genre'].unique())
    if not genre_games_present or not all_genres_present: logging.warning("⚠️ No valid games or genres left after processing. Skipping."); return None, None

    genre_game_lookup = {game: i for i, game in enumerate(genre_games_present)} # Game ID -> Sim Matrix Index
    genre_df_filtered['game_cat_genre'] = genre_df_filtered['game_id'].map(genre_game_lookup)
    genre_to_index = {genre: i for i, genre in enumerate(all_genres_present)}
    genre_df_filtered['genre_cat'] = genre_df_filtered['genre'].map(genre_to_index)
    genre_df_filtered.dropna(subset=['game_cat_genre', 'genre_cat'], inplace=True)
    if genre_df_filtered.empty: logging.warning("⚠️ No valid game-genre pairs after mapping. Skipping."); return None, None
    genre_df_filtered['game_cat_genre'] = genre_df_filtered['game_cat_genre'].astype(int)
    genre_df_filtered['genre_cat'] = genre_df_filtered['genre_cat'].astype(int)

    shape = (len(genre_game_lookup), len(all_genres_present)) # games_with_genres x unique_genres
    game_genre_matrix = sparse.csr_matrix((np.ones(len(genre_df_filtered)), (genre_df_filtered['game_cat_genre'], genre_df_filtered['genre_cat'])), shape=shape, dtype=np.float32)
    logging.info(" Normalizing genre matrix and calculating similarity...")
    try:
        normalized_matrix = normalize(game_genre_matrix, norm='l2', axis=1)
        similarity_matrix = normalized_matrix.dot(normalized_matrix.T).tocsr() # games_with_genres x games_with_genres
    except Exception as e: logging.error(f"❌ Error during genre matrix normalization/similarity: {e}"); return None, None
    logging.info(f"✅ Genre similarity matrix built. Shape: {similarity_matrix.shape} for {len(genre_game_lookup)} games.")
    return similarity_matrix, genre_game_lookup


# --- save_model_artifacts (Handles GPU factors) ---
def save_model_artifacts(model_dir, model, user_map, game_map, game_id_map,
                         genre_sim_matrix, genre_game_lookup, metrics=None,
                         best_hpo_params=None):
    """Saves model, mappings, matrices, and metadata. Handles GPU factors."""
    logging.info(f"💾 Saving model artifacts to {model_dir}...")
    os.makedirs(model_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        # --- Save main model object (with warning) ---
        model_path = os.path.join(model_dir, "als_model.pkl")
        try:
            with open(model_path, "wb") as f: pickle.dump(model, f)
            logging.info(f" Saved complete ALS model object to {model_path}")
        except Exception as pickle_err:
             logging.warning(f"⚠️ Could not pickle full model object (may contain GPU refs): {pickle_err}. Saving factors separately is recommended.")
             if os.path.exists(model_path): os.remove(model_path)

        # --- Save factors (Convert GPU->NumPy first) ---
        item_factors_to_save = None; user_factors_to_save = None
        if hasattr(model, 'item_factors') and model.item_factors is not None:
            if 'implicit.gpu' in str(type(model.item_factors)):
                logging.info(" Converting item_factors from GPU to NumPy for saving...")
                try: item_factors_to_save = model.item_factors.to_numpy()
                except Exception as e: logging.error(f"❌ Failed to convert item_factors from GPU: {e}")
            else: item_factors_to_save = model.item_factors # Assume NumPy compatible
            if item_factors_to_save is not None:
                 np.save(os.path.join(model_dir, "item_factors.npy"), item_factors_to_save)
                 logging.info(f" Saved item_factors.npy (shape: {item_factors_to_save.shape})")

        if hasattr(model, 'user_factors') and model.user_factors is not None:
             if 'implicit.gpu' in str(type(model.user_factors)):
                 logging.info(" Converting user_factors from GPU to NumPy for saving...")
                 try: user_factors_to_save = model.user_factors.to_numpy()
                 except Exception as e: logging.error(f"❌ Failed to convert user_factors from GPU: {e}")
             else: user_factors_to_save = model.user_factors
             if user_factors_to_save is not None:
                  np.save(os.path.join(model_dir, "user_factors.npy"), user_factors_to_save)
                  logging.info(f" Saved user_factors.npy (shape: {user_factors_to_save.shape})")

        # --- Save mappings ---
        with open(os.path.join(model_dir, "user_map.pkl"), "wb") as f: pickle.dump(user_map, f)
        with open(os.path.join(model_dir, "game_map.pkl"), "wb") as f: pickle.dump(game_map, f)
        with open(os.path.join(model_dir, "game_id_map.pkl"), "wb") as f: pickle.dump(game_id_map, f)

        # --- Save genre artifacts ---
        if genre_sim_matrix is not None and genre_game_lookup is not None:
             sparse.save_npz(os.path.join(model_dir, "genre_sim_matrix.npz"), genre_sim_matrix)
             with open(os.path.join(model_dir, "genre_game_lookup.pkl"), "wb") as f: pickle.dump(genre_game_lookup, f)
             logging.info(f" Saved genre_sim_matrix.npz and genre_game_lookup.pkl")
        else: logging.info("⚠️ Genre similarity matrix/lookup None, skipping save.")

        # --- Save metadata ---
        metadata = {
            "timestamp": timestamp, "metrics": metrics or {}, "best_hpo_params": best_hpo_params or {},
            "model_class": type(model).__name__ if model else None,
            "final_model_params": { "factors": getattr(model, 'factors', None), "iterations": getattr(model, 'iterations', None), "regularization": getattr(model, 'regularization', None), "alpha_confidence": best_hpo_params.get("alpha_confidence") if best_hpo_params else None },
            "evaluation_k": config.EVALUATION_K, "min_playtime": config.MIN_PLAYTIME, "random_seed": config.RANDOM_SEED, "neo4j_uri": config.NEO4J_URI,
        }
        metadata_path = os.path.join(model_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f: pickle.dump(metadata, f)
        logging.info(f" Saved metadata to {metadata_path}")

        # --- Save human-readable summary ---
        summary_path = os.path.join(model_dir, "training_summary.txt")
        with open(summary_path, "w") as f:
             f.write(f"Timestamp: {timestamp}\nModel: {metadata['model_class']}\nFinal Params Used: {metadata['final_model_params']}\n")
             f.write("-" * 20 + "\nBest HPO Params:\n")
             if metadata['best_hpo_params']:
                  for k, v in metadata['best_hpo_params'].items(): f.write(f"  {k}: {v}\n")
             else: f.write("  N/A\n")
             f.write("-" * 20 + "\nEval Metrics (Full Filtered Test Set):\n")
             if metadata['metrics']:
                  for k, v in metadata['metrics'].items(): f.write(f"  {k}: {v:.4f}\n" if isinstance(v, (int, float)) else f"  {k}: {v}\n")
             else: f.write("  N/A\n")
             f.write("-" * 20 + f"\nK: {config.EVALUATION_K}, Min Playtime: {config.MIN_PLAYTIME}, Seed: {config.RANDOM_SEED}\n")
        logging.info(f" Saved human-readable training summary to {summary_path}")

        logging.info(f"✅ Model artifacts successfully saved to ./{model_dir}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to save model artifacts: {type(e).__name__} - {e}", exc_info=True)
        return False


# --- load_model_artifacts (Unchanged from previous correct version) ---
def load_model_artifacts(model_dir):
    logging.info(f"📦 Loading model artifacts from {model_dir}...")
    artifacts = {}
    required_files = { 'model': os.path.join(model_dir, "als_model.pkl"), 'game_map': os.path.join(model_dir, "game_map.pkl"), 'game_id_map': os.path.join(model_dir, "game_id_map.pkl") }
    optional_files = { 'user_map': os.path.join(model_dir, "user_map.pkl"), 'genre_sim_matrix': os.path.join(model_dir, "genre_sim_matrix.npz"), 'genre_game_lookup': os.path.join(model_dir, "genre_game_lookup.pkl"), 'metadata': os.path.join(model_dir, "metadata.pkl"), 'item_factors': os.path.join(model_dir, "item_factors.npy"), 'user_factors': os.path.join(model_dir, "user_factors.npy") }
    all_required_loaded = True
    try:
        for key, path in required_files.items():
            if not os.path.exists(path): logging.error(f"❌ Required artifact file not found: {path}"); all_required_loaded = False; artifacts[key] = None
            else:
                try:
                    if path.endswith(".pkl"):
                        with open(path, "rb") as f: artifacts[key] = pickle.load(f)
                    elif path.endswith(".npy"): artifacts[key] = np.load(path)
                    logging.debug(f" Loaded required artifact: {key} from {path}")
                except Exception as e: logging.error(f"❌ Failed to load required artifact {key} from {path}: {e}"); all_required_loaded = False; artifacts[key] = None
        for key, path in optional_files.items():
            if os.path.exists(path):
                try:
                    if path.endswith(".pkl"):
                        with open(path, "rb") as f: artifacts[key] = pickle.load(f)
                    elif path.endswith(".npz"): artifacts[key] = sparse.load_npz(path)
                    elif path.endswith(".npy"): artifacts[key] = np.load(path)
                    logging.debug(f" Loaded optional artifact: {key} from {path}")
                except Exception as e: logging.warning(f"⚠️ Could not load optional artifact {key} from {path}: {e}"); artifacts[key] = None
            else: logging.info(f" Optional artifact file not found, skipping: {path}"); artifacts[key] = None
        # Populate factors from model object if possible
        if 'model' in artifacts and artifacts['model'] is not None:
             if ('item_factors' not in artifacts or artifacts['item_factors'] is None) and hasattr(artifacts['model'], 'item_factors'): artifacts['item_factors'] = artifacts['model'].item_factors
             if ('user_factors' not in artifacts or artifacts['user_factors'] is None) and hasattr(artifacts['model'], 'user_factors'): artifacts['user_factors'] = artifacts['model'].user_factors
        if not all_required_loaded: logging.error("❌ Failed to load required artifacts. Inference may fail."); return artifacts # Return partial
        logging.info("✅ Model artifacts loaded successfully."); return artifacts
    except Exception as e: logging.error(f"❌ Unexpected error during artifact loading: {e}", exc_info=True); return None