#!/usr/bin/env python3
"""
Example script demonstrating how to use the Steam Recommender system.
This script shows how to:
1. Initialize the recommender
2. Get hybrid recommendations for a Steam ID or vanity URL
"""

import logging
import json
# Import the correct class
from als_hybrid_recommender import ALSHybridRecommender # Use the actual class name

# --- Basic Logging Config (Root Logger) ---
logging.basicConfig(
    level=logging.WARNING, # Set default level higher (e.g., WARNING)
    format='%(levelname)s: %(message)s'
)
# --- Configure Specific Loggers ---
# Set the logger for the recommender class to ERROR to hide its INFO and WARNING messages
logging.getLogger('als_hybrid_recommender').setLevel(logging.ERROR)
# Set the logger for the main script (optional)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Keep INFO for this script's feedback if needed, like errors

# --- Import config (needed for DEFAULT_RECOMMENDATIONS) ---
try: import config
except ImportError:
    # Define minimal defaults if config is missing
    class ConfigMock: DEFAULT_RECOMMENDATIONS = 10
    config = ConfigMock()
# ---

def main():
    """Main function demonstrating the Steam Recommender."""
    print("Initializing Recommender...") # User feedback
    recommender = ALSHybridRecommender(model_dir='model')

    if recommender.model is None:
         logging.error("Recommender initialization failed (model/mappings not loaded). Exiting.")
         if recommender.driver: recommender.close()
         return

    try:
        steam_input = input("Enter Steam ID, vanity URL, or full profile URL: ")
        default_n = getattr(config, 'DEFAULT_RECOMMENDATIONS', 10)
        try:
            n_input = input(f"How many recommendations? (default: {default_n}): ")
            num_recs = int(n_input) if n_input else default_n
            if num_recs <= 0:
                 logging.warning("Number of recommendations must be positive. Using default.")
                 num_recs = default_n
        except ValueError:
            logging.warning("Invalid number input. Using default.")
            num_recs = default_n

        # Removed the prompt asking about comparison

        print(f"\nFetching {num_recs} Recommendations for '{steam_input}'...") # User feedback
        # Get hybrid recommendations directly
        recommendations = recommender.get_hybrid_recommendations(steam_input, n=num_recs)

        # --- Print Recommendations ---
        print("\n--- Recommendations ---") # Simplified title
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                game_name = rec.get('name', f"Game {rec['appid']}")
                source_raw = rec.get('source', 'unknown')
                source_display = f"[{source_raw.upper()}]"
                score = rec.get('score', 0.0); explanation = rec.get('explanation', 'N/A')

                print(f"{i}. {source_display} {game_name} (Score: {score:.3f})")
                print(f"   Steam URL: https://store.steampowered.com/app/{rec['appid']}")
                # Only print reason if it's not the generic ALS one
                if source_raw.upper() != 'ALS':
                    print(f"   Reason: {explanation}")
                genres = rec.get('genres', [])
                if genres: print(f"   Genres: {', '.join(genres)}")
                print("-" * 10) # Separator
        else:
            print(f"Could not generate recommendations for '{steam_input}'.")

        # --- Comparison code was removed ---

        # --- Optional Extra Examples section removed ---

    except KeyboardInterrupt: print("\nOperation interrupted by user.")
    except Exception as e: logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if recommender and recommender.driver: recommender.close()

if __name__ == "__main__":
    main()