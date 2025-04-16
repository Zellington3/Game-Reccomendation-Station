import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from typing import List, Dict, Tuple, Optional, Union
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from implicit.evaluation import ndcg_at_k, precision_at_k
from scipy.sparse import csr_matrix
from neo4j import GraphDatabase, basic_auth

import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class SteamRecommender:
    """A class for generating game recommendations using hybrid collaborative filtering."""
    
    def __init__(self, model_dir: str = "model"):
        """Initialize the recommender with the path to the model directory."""
        self.model_dir = model_dir
        self.model = None
        self.user_mapping = None
        self.item_mapping = None
        self.game_id_map = None  
        self.game_metadata = None 
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self.load_model()
        
    def _fetch_game_metadata(self, app_ids: List[str]) -> Dict:
        """Fetch game metadata from Neo4j database."""
        import time
        
        metadata = {}
        
        cache_file = os.path.join(self.model_dir, 'game_metadata_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_metadata = json.load(f)
                    # Check if we have all the requested app_ids
                    if all(app_id in cached_metadata for app_id in app_ids):
                        logger.info("Using cached game metadata")
                        return cached_metadata
            except Exception as e:
                logger.warning(f"Error reading metadata cache: {e}")
        # If not in cache, fetch from Neo4j
        try:
            with self.driver.session(database="neo4j") as session:
                # Query to get game metadata - simplified to match actual schema
                query = """
                MATCH (g:Game)
                WHERE g.app_id IN $app_ids
                RETURN g.app_id as app_id,
                       g.name as name,
                       [x IN [(g)-[:HAS_GENRE]->(ge) | ge.name] | x] as genres
                """
                
                result = session.run(query, app_ids=app_ids)
                
                for record in result:
                    app_id = record["app_id"]
                    metadata[app_id] = {
                        'name': record["name"],
                        'img_icon_url': '',  # Default empty values for missing properties
                        'img_logo_url': '',
                        'has_community_visible_stats': False,
                        'genres': record["genres"],
                        'categories': [],  # Default empty list for missing relationships
                        'short_description': '',
                        'release_date': '',
                        'developer': [],
                        'publisher': []
                    }
                
                # Cache the metadata
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"Cached metadata for {len(metadata)} games")
                except Exception as e:
                    logger.warning(f"Failed to cache metadata: {e}")
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error fetching metadata from Neo4j: {e}")
            # Fallback to Steam API if Neo4j fails
            logger.info("Falling back to Steam API for metadata")
            return self._fetch_game_metadata_from_steam(app_ids)
        finally:
            if 'driver' in locals():
                driver.close()
                logger.info("Neo4j connection closed")

    def _fetch_game_metadata_from_steam(self, app_ids: List[str]) -> Dict:
        """Fallback method to fetch game metadata from Steam's Store API."""
        import time
        
        metadata = {}
        
        # Process in smaller batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(app_ids), batch_size):
            batch = app_ids[i:i + batch_size]
            logger.info(f"Fetching metadata from Steam API for batch {i//batch_size + 1}/{(len(app_ids) + batch_size - 1)//batch_size}")
            
            for app_id in batch:
                try:
                    # Use the Steam Store API which is more reliable
                    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
                    response = requests.get(url)
                    data = response.json()
                    
                    if data.get(app_id, {}).get('success', False):
                        details = data[app_id]['data']
                        metadata[app_id] = {
                            'name': details.get('name', f'Game {app_id}'),
                            'img_icon_url': details.get('header_image', ''),
                            'img_logo_url': details.get('header_image', ''),
                            'has_community_visible_stats': details.get('has_community_visible_stats', False),
                            'genres': [g.get('description', '') for g in details.get('genres', [])],
                            'categories': [c.get('description', '') for c in details.get('categories', [])],
                            'short_description': details.get('short_description', ''),
                            'release_date': details.get('release_date', {}).get('date', ''),
                            'developer': details.get('developers', []),
                            'publisher': details.get('publishers', [])
                        }
                        logger.debug(f"Successfully fetched metadata for app {app_id}")
                    else:
                        logger.warning(f"Failed to fetch metadata for app {app_id}: {data.get(app_id, {}).get('error', 'Unknown error')}")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error fetching metadata for app {app_id}: {e}")
                    continue
        
        logger.info(f"Fetched metadata for {len(metadata)} games from Steam API")
        return metadata

    def load_model(self) -> None:
        """Load the trained model and mappings from disk."""
        try:
            # Updated file paths to match what's in the model directory
            model_path = os.path.join(self.model_dir, 'als_model_fixed.pkl')
            user_mapping_path = os.path.join(self.model_dir, 'user_map.pkl')
            item_mapping_path = os.path.join(self.model_dir, 'game_map.pkl')
            game_id_map_path = os.path.join(self.model_dir, 'game_id_map.pkl')
            metadata_path = os.path.join(self.model_dir, 'steam_metadata.json')
            
            # Check if the fixed model file exists, if not, try the original model file
            if not os.path.exists(model_path):
                original_model_path = os.path.join(self.model_dir, 'als_model.pkl')
                if os.path.exists(original_model_path):
                    logger.warning(f"Fixed model file not found at {model_path}. Using original model file.")
                    model_path = original_model_path
                else:
                    missing_files = [p for p in [model_path, user_mapping_path, item_mapping_path, game_id_map_path] if not os.path.exists(p)]
                    raise FileNotFoundError(f"Model files not found: {missing_files}. Please train the model first.")
            
            # Load and verify the model
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded. Type: {type(self.model)}")
            except (AttributeError, ImportError) as e:
                logger.warning(f"Error loading model with GPU support: {e}")
                logger.warning("Attempting to load model in CPU mode...")
                
                # Create a new CPU model with the same parameters
                from implicit.als import AlternatingLeastSquares
                
                # Try to extract parameters from the saved model
                try:
                    with open(model_path, 'rb') as f:
                        saved_model = pickle.load(f)
                    
                    # Create a new CPU model
                    self.model = AlternatingLeastSquares(
                        factors=getattr(saved_model, 'factors', 64),
                        regularization=getattr(saved_model, 'regularization', 0.01),
                        iterations=getattr(saved_model, 'iterations', 20),
                        random_state=getattr(saved_model, 'random_state', 42),
                        use_gpu=False  # Force CPU mode
                    )
                    
                    # Copy the factors if available
                    if hasattr(saved_model, 'user_factors') and hasattr(saved_model, 'item_factors'):
                        self.model.user_factors = saved_model.user_factors
                        self.model.item_factors = saved_model.item_factors
                        logger.info("Successfully loaded model in CPU mode with factors")
                    else:
                        logger.warning("Could not copy factors from saved model")
                except Exception as e2:
                    logger.error(f"Failed to load model in CPU mode: {e2}")
                    raise
            
            # Load and verify user mapping
            with open(user_mapping_path, 'rb') as f:
                self.user_mapping = pickle.load(f)
            logger.info(f"User mapping loaded with {len(self.user_mapping)} users")
            
            # Load and verify item mapping
            with open(item_mapping_path, 'rb') as f:
                self.item_mapping = pickle.load(f)
                # Convert all game IDs to strings
                self.item_mapping = {str(game_id): idx for game_id, idx in self.item_mapping.items()}
            logger.info(f"Item mapping loaded with {len(self.item_mapping)} items")
            
            # Load and verify game ID map (reverse mapping)
            with open(game_id_map_path, 'rb') as f:
                self.game_id_map = pickle.load(f)
                # Convert all indices to strings
                self.game_id_map = {str(idx): game_id for idx, game_id in self.game_id_map.items()}
            logger.info(f"Game ID map loaded with {len(self.game_id_map)} items")
            
            # Load game metadata from cached file
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.game_metadata = json.load(f)
                logger.info(f"Game metadata loaded with {len(self.game_metadata)} games")
                
                # Check if we need to fetch metadata for any missing games
                app_ids = list(self.item_mapping.keys())
                missing_app_ids = [app_id for app_id in app_ids if app_id not in self.game_metadata]
                
                if missing_app_ids:
                    logger.warning(f"Found {len(missing_app_ids)} games without metadata. Fetching from Steam API...")
                    new_metadata = self._fetch_game_metadata_from_steam(missing_app_ids)
                    
                    # Update existing metadata with new data
                    self.game_metadata.update(new_metadata)
                    
                    # Save updated metadata
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(self.game_metadata, f, indent=2)
                    
                    logger.info(f"Updated game metadata with {len(self.game_metadata)} games")
            else:
                logger.warning(f"Game metadata file not found at {metadata_path}")
                self.game_metadata = {}
                
                # Fetch metadata from Steam API
                logger.info("Fetching game metadata from Steam API")
                app_ids = list(self.item_mapping.keys())
                self.game_metadata = self._fetch_game_metadata_from_steam(app_ids)
                
                # Save metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.game_metadata, f, indent=2)
                
                logger.info(f"Saved game metadata with {len(self.game_metadata)} games")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def get_steam_id(self, steam_id_or_vanity: str) -> str:
        """
        Convert a Steam vanity URL to a numeric Steam ID.
        
        Args:
            steam_id_or_vanity: Steam ID or vanity URL
            
        Returns:
            Numeric Steam ID
        """
        logger.info(f"Resolving Steam ID for: {steam_id_or_vanity}")
        
        # Convert to string if it's a numeric type
        steam_id_or_vanity = str(steam_id_or_vanity)
        
        # If it's already a numeric ID, return it
        if steam_id_or_vanity.isdigit():
            logger.info(f"Input is already a numeric Steam ID: {steam_id_or_vanity}")
            return steam_id_or_vanity
            
        # If it's a full URL, extract the ID or vanity name
        if "steamcommunity.com" in steam_id_or_vanity:
            logger.info("Detected full Steam profile URL, extracting ID or vanity name")
            parts = steam_id_or_vanity.split("/")
            if "profiles" in parts:
                # It's a numeric ID
                idx = parts.index("profiles")
                if idx + 1 < len(parts):
                    numeric_id = parts[idx + 1]
                    logger.info(f"Extracted numeric Steam ID from URL: {numeric_id}")
                    return numeric_id
            elif "id" in parts:
                # It's a vanity URL
                idx = parts.index("id")
                if idx + 1 < len(parts):
                    vanity_name = parts[idx + 1]
                    logger.info(f"Extracted vanity name from URL: {vanity_name}")
                    steam_id_or_vanity = vanity_name
            
        # Otherwise, resolve the vanity URL
        if not config.STEAM_API_KEY:
            logger.error("Steam API key not found. Please set the STEAM_API_KEY environment variable.")
            raise ValueError("Steam API key not found. Please set the STEAM_API_KEY environment variable.")
            
        logger.info(f"Resolving vanity URL: {steam_id_or_vanity}")
        url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/?key={config.STEAM_API_KEY}&vanityurl={steam_id_or_vanity}"
        response = requests.get(url)
        data = response.json()
        
        if data['response']['success'] != 1:
            logger.error(f"Failed to resolve Steam vanity URL: {steam_id_or_vanity}")
            logger.error(f"API response: {data}")
            raise ValueError(f"Could not resolve Steam vanity URL: {steam_id_or_vanity}")
            
        steam_id = data['response']['steamid']
        logger.info(f"Successfully resolved vanity URL to Steam ID: {steam_id}")
        return steam_id
        
    def get_user_games(self, steam_id: str) -> List[Dict]:
        """Get games that a user has played, liked, or had recommended to them."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {steam_id: $steam_id})-[r:PLAYED|:LIKED|:RECOMMENDED]->(g:Game)
                RETURN g.app_id as app_id, 
                       g.name as name,
                       type(r) as interaction_type,
                       g.time_to_finish_h as time_to_finish,
                       g.time_to_complete_h as time_to_complete
                ORDER BY g.time_to_finish_h DESC
            """, steam_id=steam_id)
            return [dict(record) for record in result]
    
    def get_similar_games(self, game_name: str, limit: int = 10) -> List[Dict]:
        """Get similar games based on name similarity and completion times."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (g:Game)
                WHERE g.name_steam_games_database = $game_name
                MATCH (g)-[s:SIMILAR_TO]->(similar:Game)
                WHERE s.similarity > 0.7
                RETURN similar.app_id as app_id,
                       similar.name as name,
                       s.similarity as similarity,
                       similar.time_to_finish_h as time_to_finish,
                       similar.time_to_complete_h as time_to_complete
                ORDER BY s.similarity DESC
                LIMIT $limit
            """, game_name=game_name, limit=limit)
            return [dict(record) for record in result]
    
    def get_popular_games(self, limit: int = 10) -> List[Dict]:
        """Get popular games based on user interactions."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (g:Game)<-[r:PLAYED|:LIKED|:RECOMMENDED]-(u:User)
                WITH g, count(r) as interaction_count
                WHERE interaction_count > 0
                RETURN g.app_id as app_id,
                       g.name as name,
                       interaction_count,
                       g.time_to_finish_h as time_to_finish,
                       g.time_to_complete_h as time_to_complete
                ORDER BY interaction_count DESC
                LIMIT $limit
            """, limit=limit)
            return [dict(record) for record in result]
    
    def get_recommendations(self, steam_id: str, n: int = 10) -> List[Dict]:
        """Get personalized game recommendations for a user."""
        # Get user's games
        user_games = self.get_user_games(steam_id)
        if not user_games:
            logger.info("No user games found, returning popular games")
            return self.get_popular_games(n)
        
        # Get similar games for each game the user has played
        similar_games = []
        for game in user_games:
            similar = self.get_similar_games(game['name'], limit=n)
            similar_games.extend(similar)
        
        # Remove duplicates and games the user already has
        user_app_ids = {game['app_id'] for game in user_games}
        unique_recommendations = {}
        
        for game in similar_games:
            if game['app_id'] not in user_app_ids:
                if game['app_id'] not in unique_recommendations:
                    unique_recommendations[game['app_id']] = game
                else:
                    # If we've seen this game before, take the higher similarity score
                    if game['similarity'] > unique_recommendations[game['app_id']]['similarity']:
                        unique_recommendations[game['app_id']] = game
        
        # Sort by similarity and return top N
        recommendations = list(unique_recommendations.values())
        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        
        # If we don't have enough recommendations, add popular games
        if len(recommendations) < n:
            popular_games = self.get_popular_games(n - len(recommendations))
            for game in popular_games:
                if game['app_id'] not in user_app_ids and game['app_id'] not in unique_recommendations:
                    recommendations.append(game)
        
        return recommendations[:n]
    
    def close(self):
        self.driver.close()

def main():
    recommender = SteamRecommender()
    try:
        steam_id = input("Enter your Steam ID: ")
        n = int(input("How many recommendations do you want? (default: 10): ") or "10")
        
        recommendations = recommender.get_recommendations(steam_id, n)
        
        print("\nRecommended Games:")
        for i, game in enumerate(recommendations, 1):
            print(f"\n{i}. {game['name']}")
            print(f"   Similarity: {game.get('similarity', 'N/A')}")
            print(f"   Time to finish: {game.get('time_to_finish', 'N/A')} hours")
            print(f"   Time to complete: {game.get('time_to_complete', 'N/A')} hours")
            
    finally:
        recommender.close()

if __name__ == "__main__":
    main() 
