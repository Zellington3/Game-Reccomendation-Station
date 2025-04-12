import logging
import optuna
import numpy as np
import os 
from scipy.sparse import csr_matrix 

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, ndcg_at_k

# Setup logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.enable_propagation()  
optuna.logging.disable_default_handler() # Avoid duplicate logs

import config
import data_loader
import utils


def objective(trial, train_matrix_orig, test_matrix, k, random_state):
    # Optuna objective function for tuning ALE hyperparms
    factors = trial.suggest_int("factors", 32, 256, log=True)
    # Regularization
    regularization = trial.suggest_float("regularization", 1e-3, 1.0, log=True) 
    iterations = trial.suggest_int("iterations", 10, 60) 
    # Alpha scaling factor for confidence: C = 1 + alpha * R
    # A higher alpha gives more weight to higher interaction values (e.g., longer playtime)
    alpha_confidence = trial.suggest_float("alpha_confidence", 1.0, 60.0, log=True) # Common range, adjust as needed

    trial_params = {
        "factors": factors,
        "regularization": regularization,
        "iterations": iterations,
        "alpha_confidence": alpha_confidence
    }
    logging.info(f"[Trial {trial.number}] Suggesting Params: {trial_params}")

    # --- Apply alpha scaling to the training matrix for this trial ---
    # Calculate confidence: C = 1 + alpha * R
    # Ensure the matrix remains sparse and float
    train_matrix_scaled = train_matrix_orig.astype(np.float32) * alpha_confidence
    train_matrix_scaled.data += 1.0 # Efficiently add 1 only to non-zero elements
    # ------------------------------------------------------------

    # Train model
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
        calculate_training_loss=False
    )

    try:
        # Implicit ALS expects items x users for fit()
        # Pass the *scaled* train_matrix (users x items) with confidence scores
        # Ensure it's CSR format as preferred by fit
        model.fit(train_matrix_scaled.T.tocsr(), show_progress=False) # Use scaled matrix

        # Evaluate using NDCG@K (primary metric)
        # Evaluation uses the *original interaction matrices (train/test)
        # to know *which* items users interacted with (ground truth)
        current_ndcg = ndcg_at_k(
            model,
            train_matrix_orig.tocsr(), # Original train matrix for filtering seen items
            test_matrix.tocsr(),     # Test matrix for evaluation
            K=k,
            show_progress=False,
            num_threads=1 # Dont change needed for optuna 
        )
        logging.info(f"[Trial {trial.number}] Alpha: {alpha_confidence:.2f} -> NDCG@{k}: {current_ndcg:.4f}")

        return current_ndcg 

    except Exception as e:
         logging.error(f"[Trial {trial.number}] Failed with params {trial_params}: {e}", exc_info=True) # Add traceback
         return -1.0 # Indicates failure


if __name__ == "__main__":
    logging.info("🚀 Starting Training Pipeline...")

    # Configuration Defaults (Make sure  config.py or defined here)
    # Example: Add default alpha if not found after tuning
    DEFAULT_ALS_ALPHA = getattr(config, 'DEFAULT_ALS_ALPHA', 40.0) # adjust based on data scale

    # 1 - Data load 
    likes_df, genre_df = data_loader.load_neo4j_data(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password_file=config.DB_PASSWORD_FILE,
        min_playtime=config.MIN_PLAYTIME
    )

    if likes_df.empty:
        logging.error("❌ No interaction data loaded or available after filtering. Cannot proceed with training. Exiting.")
        exit(1)

    # 2 - Prepare Sparse Matrices (Train/Test Split, Mappings)
    matrix_results = utils.create_sparse_matrices(
        likes_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    if matrix_results is None:
        logging.error("❌ Failed to create sparse matrices. Exiting.")
        exit(1)
    train_matrix, test_matrix, user_map, game_map, game_id_map = matrix_results
    logging.info(f"Train matrix shape: {train_matrix.shape}, Non-zero entries: {train_matrix.nnz}")
    logging.info(f"Test matrix shape: {test_matrix.shape}, Non-zero entries: {test_matrix.nnz}")


    # 3 - Hyperparameter Tuning with Optuna
    logging.info(f"⚙️ Starting hyperparameter tuning ({config.OPTUNA_N_TRIALS} trials, timeout={config.OPTUNA_TIMEOUT}s)...")
    study = optuna.create_study(direction="maximize", study_name="als_tuning_with_alpha")
    try:
        # Pass the original train_matrix to the objective function
        study.optimize(
            lambda trial: objective(trial, train_matrix, test_matrix, config.EVALUATION_K, config.RANDOM_SEED),
            n_trials=config.OPTUNA_N_TRIALS,
            timeout=config.OPTUNA_TIMEOUT, # Optional timeout
            show_progress_bar=True # Show progress
        )
        best_params = study.best_params
        best_value = study.best_value
        logging.info(f"🏆 Optuna tuning finished. Best Trial {study.best_trial.number} NDCG@{config.EVALUATION_K}: {best_value:.4f}")
        logging.info(f" Best Params found: {best_params}")

    except KeyboardInterrupt:
         logging.warning(" Optuna study interrupted by user.")
         if study.best_trial:
              best_params = study.best_trial.params
              logging.info(f" Using best parameters found before interruption: {best_params}")
         else:
              best_params = {
                   "factors": config.DEFAULT_ALS_FACTORS,
                   "regularization": config.DEFAULT_ALS_REGULARIZATION,
                   "iterations": config.DEFAULT_ALS_ITERATIONS,
                   "alpha_confidence": DEFAULT_ALS_ALPHA # Use default alpha
              }
              logging.info(f" No successful trials completed. Falling back to default parameters: {best_params}")
    except Exception as e:
         logging.error(f"❌ Optuna study failed unexpectedly: {e}. Falling back to default parameters.", exc_info=True)
         best_params = {
             "factors": config.DEFAULT_ALS_FACTORS,
             "regularization": config.DEFAULT_ALS_REGULARIZATION,
             "iterations": config.DEFAULT_ALS_ITERATIONS,
             "alpha_confidence": DEFAULT_ALS_ALPHA # Use default alpha
         }

    # Alpha_confidence is in best_params, falling back to default if needed
    if "alpha_confidence" not in best_params:
        logging.warning(f"alpha_confidence not found in best_params, using default: {DEFAULT_ALS_ALPHA}")
        best_params["alpha_confidence"] = DEFAULT_ALS_ALPHA

    # 4. Train Final Model with Best Parameters
    logging.info(f"🎯 Training final ALS model with selected parameters: {best_params}")

    # --- Apply alpha scaling using the best alpha for the final training ---
    best_alpha = best_params["alpha_confidence"]
    logging.info(f"Applying best alpha ({best_alpha:.2f}) scaling to the full training data...")
    train_matrix_scaled_final = train_matrix.astype(np.float32) * best_alpha
    train_matrix_scaled_final.data += 1.0
    # ------------------------------------------------------------------------

    final_model = AlternatingLeastSquares(
        factors=best_params["factors"],
        regularization=best_params["regularization"],
        iterations=best_params["iterations"],
        random_state=config.RANDOM_SEED,
        calculate_training_loss=True # Can calculate loss for final model
    )
    # Fit on the scaled full training data (items x users)
    final_model.fit(train_matrix_scaled_final.T.tocsr(), show_progress=True) # Use scaled matrix
    logging.info("✅ Final model training complete.")

    # 5. Evaluate Model
    logging.info(f"📈 Evaluating final model on test set with K={config.EVALUATION_K}...")
    try:
        # Uses more threads for final evaluation if machine allows
        num_eval_threads = os.cpu_count() // 2 if os.cpu_count() else 1
        num_eval_threads = max(1, num_eval_threads) # Ensure at least 1 thread

        # Evaluation still uses original matrices for ground truth
        final_ndcg = ndcg_at_k(
            final_model,
            train_matrix.tocsr(), # Original train matrix for filtering
            test_matrix.tocsr(),  # Original test matrix for evaluation
            K=config.EVALUATION_K,
            show_progress=True,
            num_threads=num_eval_threads
        )
        final_precision = precision_at_k(
            final_model,
            train_matrix.tocsr(), # Original train matrix for filtering
            test_matrix.tocsr(),  # Original test matrix for evaluation
            K=config.EVALUATION_K,
            show_progress=True,
            num_threads=num_eval_threads
        )

        metrics = {
            f"NDCG@{config.EVALUATION_K}": final_ndcg,
            f"Precision@{config.EVALUATION_K}": final_precision,
        }
        logging.info(f"Final Metrics: NDCG={final_ndcg:.4f}, Precision={final_precision:.4f}")
    except Exception as e:
         logging.error(f"❌ Error during final model evaluation: {e}", exc_info=True)
         metrics = None

    # 6. Similarity Matrix
    genre_sim_matrix, genre_game_lookup = utils.build_genre_similarity_matrix(genre_df, game_map)

    # 7. Save Artifacts
    success = utils.save_model_artifacts(
        model_dir=config.MODEL_DIR,
        model=final_model, # Pass the actual model object
        user_map=user_map,
        game_map=game_map,
        game_id_map=game_id_map,
        genre_sim_matrix=genre_sim_matrix,
        genre_game_lookup=genre_game_lookup,
        metrics=metrics,
        best_params=best_params # Save the parameters used (including alpha)
    )
    if success:
        logging.info("✅ Training pipeline finished successfully!")
    else:
        logging.error("❌ Training pipeline finished with errors during artifact saving.")