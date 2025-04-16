# train.py
import logging
import optuna
import numpy as np
import os
import pandas as pd
import pickle
import time
from scipy.sparse import csr_matrix
import argparse

# Assuming implicit is installed and available (CPU or GPU version)
try:
    from implicit.als import AlternatingLeastSquares
    from implicit.evaluation import precision_at_k, ndcg_at_k
    # Check if GPU version is likely installed
    try:
        import implicit.gpu
        GPU_ENABLED = True
    except ImportError:
        GPU_ENABLED = False
except ImportError:
     logging.error("❌ 'implicit' library not found. Install it (`pip install implicit` or `pip install 'implicit[gpu]'`)."); exit(1)

# --- Check for Parquet dependency ---
try:
    import pyarrow
    PARQUET_ENABLED = True
except ImportError:
    logging.error("❌ 'pyarrow' library not found. Install it (`pip install pyarrow`) for data caching.")
    PARQUET_ENABLED = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
optuna.logging.get_logger("optuna").setLevel(logging.WARNING)

# --- Import project modules ---
try:
    import config
    import data_loader
    import utils
except ImportError as e:
     logging.error(f"❌ Failed to import project modules (config, data_loader, utils): {e}"); exit(1)

# --- Objective function (Using direct use_gpu=False for eval) ---
def objective(trial, train_matrix_hpo, test_matrix_hpo, k, random_state):
    # Suggest hyperparameters
    factors = trial.suggest_int("factors", 16, 128)
    regularization = trial.suggest_float("regularization", 1e-3, 1.0, log=True)
    iterations = trial.suggest_int("iterations", 10, 60)
    alpha_confidence = trial.suggest_float("alpha_confidence", 1.0, 100.0, log=True)

    trial_params = {
        "factors": factors, "regularization": regularization,
        "iterations": iterations, "alpha_confidence": alpha_confidence
    }
    logging.info(f"[Trial {trial.number}] Suggesting Params: {trial_params}")

    # Prepare training data with confidence scaling
    train_matrix_scaled_ones = train_matrix_hpo.copy()
    train_matrix_scaled_ones.data = np.ones_like(train_matrix_scaled_ones.data)
    train_matrix_scaled = train_matrix_scaled_ones.astype(np.float32) * alpha_confidence
    train_matrix_scaled.data += 1.0

    # Initialize model (let implicit decide GPU/CPU for fit)
    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, iterations=iterations,
        random_state=random_state, calculate_training_loss=False
    )

    try:
        logging.debug(f"[Trial {trial.number}] Fitting HPO model...")
        model.fit(train_matrix_scaled.T.tocsr(), show_progress=False)

        if not hasattr(model, 'user_factors') or not hasattr(model, 'item_factors'):
            raise ValueError("Model factors missing after fit.")

        # Evaluate using original matrices
        train_csr_hpo = train_matrix_hpo.tocsr()
        test_csr_hpo = test_matrix_hpo.tocsr()

        # Dimension Check
        if model.user_factors.shape[0] != train_csr_hpo.shape[1]:
             raise ValueError(f"Dimension Mismatch! Model Item Vectors (user_factors) Rows: {model.user_factors.shape[0]} != Train Matrix Cols: {train_csr_hpo.shape[1]}.")

        # Call evaluation function forcing CPU path directly
        current_ndcg = ndcg_at_k(
            model, train_csr_hpo, test_csr_hpo, K=k,
            show_progress=False, num_threads=1,
            use_gpu=False # Force CPU path for evaluation function
        )

        logging.info(f"[Trial {trial.number}] NDCG@{k}: {current_ndcg:.4f}")
        return current_ndcg

    except MemoryError:
        logging.warning(f"[Trial {trial.number}] MemoryError during fit/eval. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        logging.error(f"[Trial {trial.number}] Failed during fit/eval: {e}", exc_info=True)
        return -1.0

# Main Execution Block 
if __name__ == "__main__":
    # Argument Parser for Cache Control
    parser = argparse.ArgumentParser(description="Train the ALS recommendation model.")
    parser.add_argument(
        '--force-reload', action='store_true',
        help="Force reloading data from Neo4j, ignoring any cached data."
    )
    args = parser.parse_args()
    # End Argument Parser

    logging.info("🚀 Starting Training Pipeline...")
    start_time_pipeline = time.time()

    # Configuration Loading
    DEFAULT_ALS_ALPHA = getattr(config, 'DEFAULT_ALS_ALPHA', 40.0)
    DEFAULT_ALS_FACTORS = getattr(config, 'DEFAULT_ALS_FACTORS', 64)
    DEFAULT_ALS_ITERATIONS = getattr(config, 'DEFAULT_ALS_ITERATIONS', 20)
    DEFAULT_ALS_REGULARIZATION = getattr(config, 'DEFAULT_ALS_REGULARIZATION', 0.01)
    OPTUNA_N_TRIALS = getattr(config, 'OPTUNA_N_TRIALS', 50)
    OPTUNA_TIMEOUT = getattr(config, 'OPTUNA_TIMEOUT', 18000)
    SAMPLE_FRACTION_HPO = getattr(config, 'HPO_SAMPLE_FRACTION', 0.15)
    MIN_USER_INTERACTIONS = getattr(config, 'MIN_USER_INTERACTIONS', 5)
    MIN_ITEM_INTERACTIONS = getattr(config, 'MIN_ITEM_INTERACTIONS', 10)
    MAX_SAFE_FACTORS = getattr(config, 'MAX_SAFE_FACTORS', 96)

    # Cache Configuration
    CACHE_DIR = "cache"
    LIKES_CACHE_FILE = os.path.join(CACHE_DIR, "likes_df.parquet")
    GENRE_CACHE_FILE = os.path.join(CACHE_DIR, "genre_df.parquet")
    os.makedirs(CACHE_DIR, exist_ok=True)
    # End Cache Configuration 

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    hpo_params_file = os.path.join(config.MODEL_DIR, "hpo_best_params.pkl")

    logging.info("-" * 30); logging.info("STEP 1: Loading Data...")
    start_time_load = time.time()
    likes_df = None
    genre_df = None
    loaded_from_cache = False

    # 1 - Data loading
    if not args.force_reload and PARQUET_ENABLED and os.path.exists(LIKES_CACHE_FILE) and os.path.exists(GENRE_CACHE_FILE):
        logging.info(f"💾 Found cache files. Attempting to load from '{CACHE_DIR}'...")
        try:
            likes_df = pd.read_parquet(LIKES_CACHE_FILE)
            genre_df = pd.read_parquet(GENRE_CACHE_FILE)
            if not isinstance(likes_df, pd.DataFrame) or not isinstance(genre_df, pd.DataFrame): raise ValueError("Cached data not DataFrame.")
            if 'user_id' not in likes_df.columns or 'game_id' not in likes_df.columns: raise ValueError("Cached likes_df missing columns.")
            if 'game_id' not in genre_df.columns or 'genre' not in genre_df.columns: raise ValueError("Cached genre_df missing columns.")
            logging.info("✅ Successfully loaded data from cache.")
            loaded_from_cache = True
        except Exception as e:
            logging.warning(f"⚠️ Failed to load data from cache: {e}. Falling back to Neo4j.")
            likes_df, genre_df = None, None; loaded_from_cache = False

    if not loaded_from_cache:
        if args.force_reload: logging.info("🔄 Forced reload from Neo4j requested.")
        else: logging.info(f"💾 Cache not found or unusable. Loading data from Neo4j...")
        likes_df, genre_df = data_loader.load_neo4j_data(min_playtime=config.MIN_PLAYTIME)

        if PARQUET_ENABLED and likes_df is not None and not likes_df.empty and genre_df is not None:
             logging.info(f"💾 Saving loaded data to cache directory '{CACHE_DIR}'...")
             try:
                  likes_df.to_parquet(LIKES_CACHE_FILE, index=False)
                  genre_df.to_parquet(GENRE_CACHE_FILE, index=False)
                  logging.info("✅ Data successfully cached.")
             except Exception as e: logging.error(f"❌ Failed to save data to cache: {e}")
        elif not PARQUET_ENABLED: logging.warning(" Parquet support (`pyarrow`) not available. Skipping caching.")

    if likes_df is None or likes_df.empty: logging.error("❌ No interaction data loaded. Exiting."); exit(1)
    if genre_df is None: logging.warning("⚠️ Genre data is None."); genre_df = pd.DataFrame(columns=['game_id', 'genre'])

    logging.info(f" Data loading complete. Likes: {len(likes_df)}, Genres: {len(genre_df)}. Duration: {time.time() - start_time_load:.2f}s")


    # 1.5 - Interaction Thresholding 
    logging.info("-" * 30); logging.info("STEP 1.5: Applying Interaction Thresholding...")
    logging.info(f" Applying thresholds: Users >= {MIN_USER_INTERACTIONS}, Items >= {MIN_ITEM_INTERACTIONS}.")
    original_rows = len(likes_df); original_users = likes_df['user_id'].nunique(); original_items = likes_df['game_id'].nunique()
    iteration = 0
    while True:
        iteration += 1; rows_before_iter = len(likes_df)
        user_counts = likes_df.groupby('user_id').size(); item_counts = likes_df.groupby('game_id').size()
        users_to_keep = user_counts[user_counts >= MIN_USER_INTERACTIONS].index; items_to_keep = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        likes_df_filtered = likes_df[likes_df['user_id'].isin(users_to_keep) & likes_df['game_id'].isin(items_to_keep)]
        if len(likes_df_filtered) == rows_before_iter: logging.info(f" Thresholding converged after {iteration} iterations."); break
        else: likes_df = likes_df_filtered.copy()
        if iteration > 10: logging.warning(" Thresholding loop exceeded 10 iterations."); break
    final_rows = len(likes_df); final_users = likes_df['user_id'].nunique(); final_items = likes_df['game_id'].nunique()
    logging.info(f" Thresholding Complete: Final Users={final_users}, Items={final_items}, Interactions={final_rows}")
    if final_rows == 0: logging.error("❌ All data removed by thresholding!"); exit(1)


    # 2a - Create FULL Split & Maps 
    logging.info("-" * 30); logging.info("STEP 2a: Creating Full Data Splits & Maps (Post-Thresholding)...")
    start_time_split_full = time.time()
    full_split_results = utils.create_sparse_matrices(likes_df.copy(), test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, create_full_split=True)
    if full_split_results is None: logging.error("❌ Failed to create full splits/maps."); exit(1)
    full_train_matrix, full_test_matrix, user_map_full, game_map_full, game_id_map_full, _, _ = full_split_results
    logging.info(f" Full maps created: {len(user_map_full)} Users, {len(game_map_full)} Games")
    logging.info(f" Full matrices created: Train Shape={full_train_matrix.shape} (nnz={full_train_matrix.nnz}), Test Shape={full_test_matrix.shape} (nnz={full_test_matrix.nnz})")
    logging.info(f" Full split duration: {time.time() - start_time_split_full:.2f}s")


    # 2b & 2c - HPO Sampling & Matrices Prep
    logging.info("-" * 30); logging.info("STEP 2b: Sampling Interactions for HPO...")
    likes_df_hpo_sample = None
    if SAMPLE_FRACTION_HPO < 1.0 and SAMPLE_FRACTION_HPO > 0.0 and len(likes_df) > 1 :
        num_sample_rows = max(1, int(len(likes_df) * SAMPLE_FRACTION_HPO))
        logging.warning(f"⚠️ SAMPLING {SAMPLE_FRACTION_HPO*100:.1f}% ({num_sample_rows}) of filtered interactions for HPO.")
        likes_df_hpo_sample = likes_df.sample(n=num_sample_rows, random_state=config.RANDOM_SEED)
        if likes_df_hpo_sample.empty: logging.error("❌ HPO sample empty!"); exit(1)
    else: logging.info(" Using full filtered dataset for HPO."); likes_df_hpo_sample = likes_df
    logging.info("-" * 30); logging.info("STEP 2c: Creating HPO Matrices...")
    start_time_split_hpo = time.time()
    hpo_matrix_results = utils.create_sparse_matrices(likes_df_hpo_sample, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, create_full_split=False, existing_user_map=user_map_full, existing_game_map=game_map_full)
    if hpo_matrix_results is None: logging.error("❌ Failed to create HPO matrices."); exit(1)
    hpo_train_matrix, hpo_test_matrix, _, _, _ = hpo_matrix_results
    logging.info(f" HPO Train matrix: Shape={hpo_train_matrix.shape}, nnz={hpo_train_matrix.nnz}")
    logging.info(f" HPO Test matrix: Shape={hpo_test_matrix.shape}, nnz={hpo_test_matrix.nnz}")
    logging.info(f" HPO matrix creation duration: {time.time() - start_time_split_hpo:.2f}s")


    # 3 - Hyperparameter Tuning (Optuna) 
    logging.info("-" * 30); logging.info("STEP 3: Hyperparameter Optimization...")
    best_hpo_params = None
    if os.path.exists(hpo_params_file):
        try:
            with open(hpo_params_file, 'rb') as f: best_hpo_params = pickle.load(f)
            logging.info(f"✅ Loaded existing HPO parameters from {hpo_params_file}: {best_hpo_params}")
        except Exception as e: logging.warning(f"⚠️ Could not load HPO parameters: {e}. Re-running HPO."); best_hpo_params = None
    if best_hpo_params is None:
        if hpo_train_matrix.nnz == 0 or hpo_test_matrix.nnz == 0: logging.error("❌ HPO train/test matrix empty. Cannot run Optuna."); best_hpo_params = None
        else:
             start_time_hpo = time.time()
             logging.info(f"⚙️ Starting new Optuna study ({OPTUNA_N_TRIALS} trials, timeout={OPTUNA_TIMEOUT}s)...")
             study = optuna.create_study(direction="maximize", study_name=f"als_tuning_sampled_{SAMPLE_FRACTION_HPO*100:.0f}pct_filtered")
             try:
                 study.optimize(lambda trial: objective(trial, hpo_train_matrix, hpo_test_matrix, config.EVALUATION_K, config.RANDOM_SEED), n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
                 hpo_duration = time.time() - start_time_hpo; logging.info(f" Optuna study finished. Duration: {hpo_duration:.2f}s")
                 if study.best_trial and study.best_value is not None and study.best_value > -1.0:
                      best_hpo_params = study.best_params; best_value = study.best_value
                      logging.info(f"🏆 Best Trial {study.best_trial.number} NDCG@{config.EVALUATION_K}: {best_value:.4f}"); logging.info(f" Best Params: {best_hpo_params}")
                      try:
                           with open(hpo_params_file, 'wb') as f: pickle.dump(best_hpo_params, f); logging.info(f"💾 Saved best HPO parameters to {hpo_params_file}")
                      except Exception as e: logging.error(f"❌ Failed to save HPO parameters: {e}")
                 else: logging.warning(" Optuna study finished, but no valid best trial found."); best_hpo_params = None
             except KeyboardInterrupt: hpo_duration = time.time() - start_time_hpo; logging.warning(f"\n Optuna interrupted after {hpo_duration:.2f}s."); best_hpo_params = study.best_trial.params if study.trials and study.best_trial and study.best_value is not None and study.best_value > -1.0 else None
             except Exception as e: logging.error(f"❌ Optuna study failed: {e}.", exc_info=True); best_hpo_params = None
    else: logging.info(" Skipping Optuna study as parameters were loaded.")
    if best_hpo_params is None:
        best_hpo_params = { "factors": DEFAULT_ALS_FACTORS, "regularization": DEFAULT_ALS_REGULARIZATION, "iterations": DEFAULT_ALS_ITERATIONS, "alpha_confidence": DEFAULT_ALS_ALPHA }
        logging.warning(f"Falling back to default HPO parameters: {best_hpo_params}")
    if "alpha_confidence" not in best_hpo_params: best_hpo_params["alpha_confidence"] = DEFAULT_ALS_ALPHA


    # Factor Capping for Memory Safety 
    original_hpo_factors = best_hpo_params.get("factors", DEFAULT_ALS_FACTORS)
    params_for_final_training = best_hpo_params.copy()
    if original_hpo_factors > MAX_SAFE_FACTORS:
        logging.warning("-" * 30); logging.warning(f"⚠️ MEMORY SAFETY: HPO suggested {original_hpo_factors} factors."); logging.warning(f" Capping factors at {MAX_SAFE_FACTORS} for final training."); params_for_final_training["factors"] = MAX_SAFE_FACTORS; logging.warning("-" * 30)
    else: logging.info(f" Using {original_hpo_factors} factors from HPO for final training (limit: {MAX_SAFE_FACTORS}).")


    # 4. Train Final Model 
    logging.info("-" * 30); logging.info(f"STEP 4: Training Final Model (on Filtered Data)...")
    logging.info(f" Using parameters: {params_for_final_training}")
    start_time_final_train = time.time()
    alpha_final = params_for_final_training["alpha_confidence"]
    logging.info(f" Applying alpha={alpha_final:.2f} scaling to FULL filtered training data (nnz={full_train_matrix.nnz})...")
    train_matrix_ones = full_train_matrix.copy(); train_matrix_ones.data = np.ones_like(train_matrix_ones.data)
    train_matrix_scaled_final = train_matrix_ones.astype(np.float32) * alpha_final; train_matrix_scaled_final.data += 1.0
    final_model = AlternatingLeastSquares( factors=params_for_final_training["factors"], regularization=params_for_final_training["regularization"], iterations=params_for_final_training["iterations"], random_state=config.RANDOM_SEED, calculate_training_loss=True )
    logging.info(" Fitting final model...")
    original_user_factors_final = None; original_item_factors_final = None
    try:
        if train_matrix_scaled_final.nnz > 0:
             if GPU_ENABLED: logging.info("✅ Detected GPU-enabled Implicit model. Training will attempt to use GPU.")
             else: logging.warning(" CPU-only Implicit model detected. Training will use CPU.")
             final_model.fit(train_matrix_scaled_final.T.tocsr(), show_progress=True)
             final_train_duration = time.time() - start_time_final_train
             logging.info(f"✅ Final model training complete. Duration: {final_train_duration:.2f}s")
             if hasattr(final_model, 'training_loss'):
                 loss_val = final_model.training_loss
                 if isinstance(loss_val, (list, np.ndarray)) and len(loss_val) > 0: loss_val = loss_val[-1]
                 logging.info(f" Final model training loss: {loss_val}")
             if hasattr(final_model, 'user_factors'): original_user_factors_final = final_model.user_factors
             if hasattr(final_model, 'item_factors'): original_item_factors_final = final_model.item_factors
        else: logging.error("❌ Cannot train final model: FULL training matrix empty."); final_model = None
    except MemoryError: logging.error(f"❌❌❌ MEMORY ERROR during final training (Factors: {params_for_final_training['factors']}). Reduce MAX_SAFE_FACTORS."); final_model = None
    except Exception as e: logging.error(f"❌ Error during final model training: {e}", exc_info=True); final_model = None

    # 5 Evaluate Final Model (Using direct use_gpu=False parameter)
    logging.info("-" * 30); logging.info("STEP 5: Evaluating Final Model (on Filtered Data)...")
    metrics = None

    # Try block JUST for evaluation
    try:
        if final_model and original_user_factors_final is not None and original_item_factors_final is not None:
            start_time_eval = time.time()

            num_eval_threads = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
            num_eval_threads = max(1, num_eval_threads)
            # Log that we are INTENDING to use the CPU path via the parameter
            logging.info(f" Evaluating final model (K={config.EVALUATION_K}) using {num_eval_threads} threads, forcing CPU evaluation path via use_gpu=False...")

            if full_test_matrix is None or full_test_matrix.nnz == 0:
                logging.warning("⚠️ Filtered test matrix empty. Skipping evaluation.")
            else:
                logging.info(" Running evaluation functions...")
                full_train_csr_for_eval = full_train_matrix.tocsr()
                full_test_csr_for_eval = full_test_matrix.tocsr()

                # Check if test matrix has enough non-zero elements
                if full_test_csr_for_eval.nnz < 10:
                    logging.warning(f"⚠️ Test matrix has very few interactions ({full_test_csr_for_eval.nnz}). Evaluation may be unreliable.")
                
                # Add detailed logging for debugging
                logging.info(f" Train matrix shape: {full_train_csr_for_eval.shape}, nnz: {full_train_csr_for_eval.nnz}")
                logging.info(f" Test matrix shape: {full_test_csr_for_eval.shape}, nnz: {full_test_csr_for_eval.nnz}")
                logging.info(f" Model item_factors shape: {final_model.item_factors.shape}")
                logging.info(f" Model user_factors shape: {final_model.user_factors.shape}")

                # Dimension Check
                if final_model.item_factors.shape[0] != full_train_csr_for_eval.shape[1]: raise ValueError(f"Item Dimension Mismatch! Model Item Vectors (item_factors) Rows: {final_model.item_factors.shape[0]} != Train Matrix Cols: {full_train_csr_for_eval.shape[1]}.")
                if final_model.user_factors.shape[0] != full_train_csr_for_eval.shape[0]: raise ValueError(f"User Dimension Mismatch! Model User Vectors (user_factors) Rows: {final_model.user_factors.shape[0]} != Train Matrix Rows: {full_train_csr_for_eval.shape[0]}.")
                logging.info(" Dimension check passed.")

                # Call evaluation functions WITH use_gpu=False
                try:
                    final_ndcg = ndcg_at_k( final_model, full_train_csr_for_eval, full_test_csr_for_eval,
                                            K=config.EVALUATION_K, show_progress=True, num_threads=num_eval_threads,
                                            use_gpu=False # Direct parameter to force CPU eval path
                                          )
                    logging.info(f" NDCG@{config.EVALUATION_K} calculation successful: {final_ndcg:.4f}")
                except Exception as ndcg_error:
                    logging.error(f"❌ Error calculating NDCG: {ndcg_error}")
                    final_ndcg = 0.0
                
                try:
                    final_precision = precision_at_k( final_model, full_train_csr_for_eval, full_test_csr_for_eval,
                                                      K=config.EVALUATION_K, show_progress=True, num_threads=num_eval_threads,
                                                      use_gpu=False # Direct parameter to force CPU eval path
                                                    )
                    logging.info(f" Precision@{config.EVALUATION_K} calculation successful: {final_precision:.4f}")
                except Exception as precision_error:
                    logging.error(f"❌ Error calculating Precision: {precision_error}")
                    final_precision = 0.0

                metrics = { f"NDCG@{config.EVALUATION_K}": final_ndcg, f"Precision@{config.EVALUATION_K}": final_precision }
                logging.info(f"🏁 Final Metrics (Filtered Test Set): NDCG={final_ndcg:.4f}, Precision={final_precision:.4f}")

            eval_duration = time.time() - start_time_eval
            logging.info(f" Evaluation duration: {eval_duration:.2f}s")
        else:
            logging.warning(" Skipping final model evaluation as training failed or factors missing.")

    except Exception as e:
        # Log any error during evaluation itself
        logging.error(f"❌ Error during final model evaluation: {e}", exc_info=True)
        metrics = None # Ensure metrics are None if eval fails
    # No finally block needed for patching restore

    # Restore factors
    if final_model:
        if original_user_factors_final is not None: final_model.user_factors = original_user_factors_final
        if original_item_factors_final is not None: final_model.item_factors = original_item_factors_final


    # 6 Build Genre Similarity Matrix 
    logging.info("-" * 30); logging.info("STEP 6: Building Genre Similarity Matrix...")
    start_time_genre = time.time()
    genre_sim_matrix = None; genre_game_lookup = None
    if genre_df is not None and not genre_df.empty and game_map_full is not None:
        genre_sim_matrix, genre_game_lookup = utils.build_genre_similarity_matrix(genre_df, game_map_full)
        if genre_sim_matrix is not None: logging.info(f" Genre similarity matrix built. Duration: {time.time() - start_time_genre:.2f}s")
        else: logging.warning(f" Genre similarity matrix could not be built. Duration: {time.time() - start_time_genre:.2f}s")
    else: logging.warning(" Skipping genre similarity matrix build as genre_df or game_map_full is missing/empty."); genre_sim_matrix, genre_game_lookup = None, None


    # 7 Save Artifacts 
    logging.info("-" * 30); logging.info("STEP 7: Saving Artifacts...")
    start_time_save = time.time()
    if final_model:
        success = utils.save_model_artifacts( model_dir=config.MODEL_DIR, model=final_model, user_map=user_map_full, game_map=game_map_full, game_id_map=game_id_map_full, genre_sim_matrix=genre_sim_matrix, genre_game_lookup=genre_game_lookup, metrics=metrics, best_hpo_params=best_hpo_params )
        if success: logging.info(f" Artifact saving complete. Duration: {time.time() - start_time_save:.2f}s")
        else: logging.error(f"❌ Artifact saving failed. Duration: {time.time() - start_time_save:.2f}s")
    else: logging.error("❌ FINAL model training failed. Artifacts NOT saved.")


    pipeline_duration = time.time() - start_time_pipeline
    logging.info("-" * 50); logging.info(f"🏁 Training Pipeline End. Total Duration: {pipeline_duration / 60:.2f} minutes"); logging.info("-" * 50)
