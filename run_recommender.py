#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from steam_recommender import SteamRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Steam Game Recommender')
    parser.add_argument('--steam-id', type=str, required=True, 
                        help='Steam ID or vanity URL to get recommendations for')
    parser.add_argument('--num-recommendations', type=int, default=10,
                        help='Number of recommendations to return (default: 10)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare hybrid and ALS-only recommendations')
    parser.add_argument('--output-format', choices=['text', 'json'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--model-dir', type=str, default='model',
                        help='Directory containing the trained model (default: model)')
    
    args = parser.parse_args()
    
    try:
        # Initialize the recommender
        recommender = SteamRecommender(model_dir=args.model_dir)
        
        if args.compare:
            # Compare hybrid and ALS-only recommendations
            results = recommender.compare_recommendations(
                args.steam_id, 
                args.num_recommendations
            )
            
            if args.output_format == 'json':
                print(json.dumps(results, indent=2))
            else:
                print("\n=== HYBRID RECOMMENDATIONS ===")
                for i, game in enumerate(results['hybrid'], 1):
                    print(f"{i}. {game['name']} (Score: {game['score']:.4f})")
                    print(f"   Steam URL: https://store.steampowered.com/app/{game['appid']}")
                
                print("\n=== ALS-ONLY RECOMMENDATIONS ===")
                for i, game in enumerate(results['als_only'], 1):
                    print(f"{i}. {game['name']} (Score: {game['score']:.4f})")
                    print(f"   Steam URL: https://store.steampowered.com/app/{game['appid']}")
        else:
            # Get regular recommendations
            recommendations = recommender.get_recommendations(
                args.steam_id, 
                args.num_recommendations
            )
            
            if args.output_format == 'json':
                print(json.dumps(recommendations, indent=2))
            else:
                print("\n=== RECOMMENDATIONS ===")
                for i, game in enumerate(recommendations, 1):
                    print(f"{i}. {game['name']} (Score: {game['score']:.4f})")
                    print(f"   Steam URL: https://store.steampowered.com/app/{game['appid']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 