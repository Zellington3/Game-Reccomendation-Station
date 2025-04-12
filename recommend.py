# recommend.py
import logging
import argparse
import numpy as np
import scipy.sparse as sparse
import os 

# Setup logging BEFORE importing 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import config  # Assuming config.py has MODEL_DIR and DEFAULT_RECOMMEND_CONFIDENCE
    import utils   # Assuming utils.py has load_model_artifacts
except ImportError as e:
     logging.error(f"Failed to import required modules (config, utils): {e}")
     logging.error("Ensure config.py and utils.py are in the same directory or Python path.")
     exit(1) # Exit if basic modules can't be imported

def get_recommendations_for_user_input(
    input_game_ids,
    input_playtimes=None, # Optional dictionary: {game_id: playtime}
    n_recommendations=10,
    artifacts=None,
    default_confidence=config.DEFAULT_RECOMMEND_CONFIDENCE # Use value from config
    ):
    """
    Generates recommendations based on user-provided game IDs and optional playtimes.

    Args:
        input_game_ids (list): List of game IDs (strings) the user likes/played.
        input_playtimes (dict, optional): Dictionary mapping game_id to playtime.
                                         Defaults to None.
        n_recommendations (int): Number of recommendations to generate.
        artifacts (dict): Dictionary containing loaded model artifacts.
                               Must include 'model', 'game_map', 'game_id_map'.
        default_confidence (float): Confidence score for games without specific playtime.

    Returns:
        list: A list of recommended game IDs (strings), or None if critical error occurs.
              Returns an empty list if no valid recommendations can be made.
    """
    if not artifacts:
         logging.error("❌ Artifacts dictionary is missing.")
         return None
    if 'model' not in artifacts or artifacts['model'] is None:
        logging.error("❌ Trained model object not found in artifacts.")
        return None
    if 'game_map' not in artifacts or artifacts['game_map'] is None:
        logging.error("❌ Game map (ID -> Index) not found in artifacts.")
        return None
    if 'game_id_map' not in artifacts or artifacts['game_id_map'] is None:
        logging.error("❌ Game ID map (Index -> ID) not found in artifacts.")
        return None

    model = artifacts['model']
    game_map = artifacts['game_map']       # Map GameID -> Internal Index
    game_id_map = artifacts['game_id_map'] # Map Internal Index -> GameID
    n_items = model.item_factors.shape[0] # Total number of items known by the model

    known_game_indices = []
    confidences = []
    unknown_games = []
    input_games_set = set(input_game_ids) # For faster filtering later

    logging.info(f"Processing {len(input_game_ids)} input games for recommendation...")
    for game_id in input_game_ids:
        if game_id in game_map:
            game_index = game_map[game_id]
            known_game_indices.append(game_index)

            # Determine confidence score
            playtime = input_playtimes.get(game_id, None) if input_playtimes else None
            if playtime is not None and playtime > 0:
                # Use the same log transform as during training
                confidence = 1.0 + np.log1p(float(playtime))
                logging.debug(f" Game '{game_id}' (Index: {game_index}): Playtime={playtime}, Confidence={confidence:.2f}")
            else:
                # Use default confidence for selected games without playtime
                confidence = default_confidence
                logging.debug(f" Game '{game_id}' (Index: {game_index}): No playtime, Default Confidence={confidence:.2f}")
            confidences.append(confidence)
        else:
            unknown_games.append(game_id)
            logging.warning(f"⚠️ Input game ID '{game_id}' not found in model's training data (game_map). Skipping.")

    if unknown_games:
         logging.warning(f" Ignored {len(unknown_games)} unknown game IDs: {', '.join(unknown_games)}")

    if not known_game_indices:
        logging.error("❌ None of the provided input game IDs were known to the model. Cannot generate ALS recommendations.")
        # --- Potential Fallback ---
        # TODO: Implement content-based fallback using genre similarity if artifacts['genre_sim_matrix'] exists
        return [] # Return empty list for now
        # --------------------------

    # Create a sparse matrix for the user's input preferences
    # Shape: (1, n_items) - representing one user
    try:
        input_user_matrix = sparse.csr_matrix((np.array(confidences, dtype=np.float32),
                                            ([0] * len(known_game_indices), known_game_indices)),
                                            shape=(1, n_items))
        logging.info(f" Created input matrix for inference with {len(known_game_indices)} known games.")
    except Exception as e:
        logging.error(f"❌ Failed to create sparse input matrix: {e}")
        return None


    # Use model.recommend to generate recommendations based on the input matrix
    # filter_already_liked_items=True should prevent recommending the input games
    try:
        logging.debug(f"Calling model.recommend with N={n_recommendations}, recalculate_user=True")
        # Request slightly more items in case filtering removes some top ones
        raw_recs = model.recommend(userid=0, # Placeholder userid=0, inference uses user_items
                                     user_items=input_user_matrix,
                                     N=n_recommendations + len(known_game_indices), # Get more to filter
                                     filter_already_liked_items=True,
                                     recalculate_user=True) # Crucial: infers vector from user_items

        # Map internal indices back to original game_ids
        recommended_game_ids = [game_id_map[idx] for idx, score in raw_recs if idx in game_id_map]

        # Filter out any input games that might have slipped through 
        final_recommendations = [rec_id for rec_id in recommended_game_ids if rec_id not in input_games_set][:n_recommendations]

        logging.info(f" Generated {len(final_recommendations)} final recommendations.")
        return final_recommendations

    except IndexError as e:
         # Can happen if user_items matrix shape is wrong or internal issues
         logging.error(f"❌ IndexError during recommendation: {e}. Check matrix dimensions and model state.")
         return None
    except Exception as e:
        logging.error(f"❌ Unexpected error during recommendation generation: {type(e).__name__} - {e}")
        return None


# MAIN RECOMMENDATION SCRIPT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate game recommendations based on user input games.")
    parser.add_argument("game_ids", type=str, help="Comma-separated list of game IDs the user likes/played (e.g., '730,570,10').")
    parser.add_argument("-p", "--playtimes", type=str, default=None, help="Optional: Comma-separated list of playtimes (float/int) corresponding to game_ids (e.g., '150,500,20'). Must match order and count of game_ids.")
    parser.add_argument("-n", "--num_recs", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument("-dc", "--default_confidence", type=float, default=config.DEFAULT_RECOMMEND_CONFIDENCE, help="Confidence score for input games without playtime.")

    args = parser.parse_args()

    # 1. Parse Input Games
    input_game_ids_list = [gid.strip() for gid in args.game_ids.split(',') if gid.strip()]
    input_playtimes_dict = None

    if args.playtimes:
        playtimes_list = [p.strip() for p in args.playtimes.split(',') if p.strip()]
        if len(playtimes_list) == len(input_game_ids_list):
            try:
                # Ensure playtimes are floats
                input_playtimes_dict = {gid: float(p) for gid, p in zip(input_game_ids_list, playtimes_list)}
                logging.info(f" Parsed playtimes: {input_playtimes_dict}")
            except ValueError:
                 logging.error("❌ Invalid playtime values provided. Ensure they are numbers. Ignoring playtimes.")
                 input_playtimes_dict = None # Fallback to default confidence
        else:
            logging.error(f"❌ Number of playtimes ({len(playtimes_list)}) does not match number of game IDs ({len(input_game_ids_list)}). Ignoring playtimes.")
            input_playtimes_dict = None # Fallback to default confidence

    if not input_game_ids_list:
        print("\nError: No game IDs provided.")
        parser.print_help()
        exit(1)

    # 2. Load the pre-trained artifacts (including the full model)
    logging.info(f"Attempting to load artifacts from directory: {config.MODEL_DIR}")
    if not os.path.isdir(config.MODEL_DIR):
        logging.error(f"❌ Model directory '{config.MODEL_DIR}' not found. Did you run the training script (`train.py`)?")
        exit(1)

    artifacts = utils.load_model_artifacts(config.MODEL_DIR)

    if artifacts:
        # 3. Get Recommendations using the inference function
        recs = get_recommendations_for_user_input(
            input_game_ids=input_game_ids_list,
            input_playtimes=input_playtimes_dict,
            n_recommendations=args.num_recs,
            artifacts=artifacts,
            default_confidence=args.default_confidence
        )

        # 4. Print Results
        print("-" * 30) # Separator
        if recs is not None:
            if recs:
                print(f"✅ Recommendations based on input games ({len(input_game_ids_list)} provided):")
                for i, game_id in enumerate(recs):
                    print(f"  {i+1}. Game ID: {game_id}")
            else:
                 # Check if it was due to no known input games or other failure
                 if 'model' in artifacts and artifacts['model']: # Check if model loaded
                      print(f"⚠️ Could not generate recommendations based on the provided input.")
                      print(f"   This might happen if none of the input games were known during model training,")
                      print(f"   or if an internal error occurred during recommendation.")
                 # If artifacts didn't load, error was already printed
        else:
            # Critical error occurred during recommendation or loading
             print(f"❌ A critical error occurred. Check logs for details.")
        print("-" * 30) # Separator


    else:
        # Error message already printed by load_model_artifacts
        logging.error("❌ Could not load model artifacts. Ensure training completed successfully.")
        exit(1) # Exit if loading fails