#!/usr/bin/env python3
import os
import json
import logging
import time
import random
import requests
from typing import List, Dict, Set
from neo4j import GraphDatabase, basic_auth
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_active_game_ids_from_neo4j() -> List[str]:
    """Get game app_ids from Neo4j database that have user interactions."""
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            # Query to get games that have user interactions
            result = session.run("""
                MATCH (g:Game)<-[:PLAYED|:LIKED|:RECOMMENDED]-(u:User)
                WHERE g.app_id IS NOT NULL
                WITH g, count(u) as interaction_count
                WHERE interaction_count > 0
                RETURN g.app_id as app_id, interaction_count
                ORDER BY interaction_count DESC
            """)
            app_ids = [record["app_id"] for record in result]
            
        logger.info(f"Found {len(app_ids)} active games with user interactions")
        return app_ids
        
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {str(e)}")
        return []
    finally:
        driver.close()

def fetch_game_details(app_id: str) -> Dict:
    """Fetch game details using public Steam Web API."""
    base_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    try:
        # First try the public API
        response = requests.get(base_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch from public API: {response.status_code}")
        
        data = response.json()
        apps = data.get('applist', {}).get('apps', [])
        
        # Find the game in the list
        game = next((app for app in apps if str(app.get('appid')) == str(app_id)), None)
        if not game:
            raise Exception("Game not found in app list")
        
        # Now try to get additional details from the store
        store_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=us&l=en"
        store_response = requests.get(store_url, headers=headers)
        
        if store_response.status_code == 200:
            store_data = store_response.json()
            if store_data and str(app_id) in store_data and store_data[str(app_id)]['success']:
                details = store_data[str(app_id)]['data']
                return {
                    'name': details.get('name', game.get('name', '')),
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
        
        # If store API fails, return basic info
        return {
            'name': game.get('name', ''),
            'img_icon_url': '',
            'img_logo_url': '',
            'has_community_visible_stats': False,
            'genres': [],
            'categories': [],
            'short_description': '',
            'release_date': '',
            'developer': [],
            'publisher': []
        }
        
    except Exception as e:
        logger.error(f"Error fetching details for app {app_id}: {str(e)}")
        return None

def fetch_steam_metadata(app_ids: List[str], batch_size: int = 5) -> Dict:
    """Fetch game metadata from Steam Web API in batches."""
    metadata = {}
    failed_app_ids = []
    
    # Load existing metadata if it exists
    try:
        with open('model/steam_metadata.json', 'r') as f:
            metadata = json.load(f)
            logger.info(f"Loaded {len(metadata)} existing games from cache")
    except FileNotFoundError:
        pass
    
    # Filter out already processed app IDs
    remaining_app_ids = [app_id for app_id in app_ids if app_id not in metadata]
    logger.info(f"Fetching metadata for {len(remaining_app_ids)} remaining games")
    
    for i in range(0, len(remaining_app_ids), batch_size):
        batch = remaining_app_ids[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{len(remaining_app_ids)//batch_size + 1}")
        
        for app_id in batch:
            try:
                details = fetch_game_details(app_id)
                if details:
                    metadata[app_id] = details
                    logger.info(f"Successfully fetched metadata for app {app_id}")
                else:
                    failed_app_ids.append(app_id)
                    logger.warning(f"Failed to fetch metadata for app {app_id}")
                
                # Random delay between requests
                time.sleep(random.uniform(1.5, 2.5))
                
            except Exception as e:
                logger.error(f"Error processing app {app_id}: {str(e)}")
                failed_app_ids.append(app_id)
                time.sleep(random.uniform(2, 4))  # Longer delay on error
        
        # Save progress after each batch
        with open('model/steam_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved progress: {len(metadata)} games processed, {len(failed_app_ids)} failed")
        
        # Random delay between batches
        time.sleep(random.uniform(3, 5))
    
    return metadata

def save_metadata(metadata: Dict, output_file: str) -> None:
    """Save metadata to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for {len(metadata)} games to {output_file}")
    except Exception as e:
        logger.error(f"Error saving metadata to {output_file}: {e}")

def main():
    """Main function to fetch and store Steam metadata."""
    # Create model directory if it doesn't exist
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Get active game IDs from Neo4j
    app_ids = get_active_game_ids_from_neo4j()
    
    # Fetch metadata from Steam API
    metadata = fetch_steam_metadata(app_ids)
    
    # Save metadata to file
    output_file = os.path.join(model_dir, 'steam_metadata.json')
    save_metadata(metadata, output_file)
    
    logger.info("Metadata fetching completed")

if __name__ == "__main__":
    main() 