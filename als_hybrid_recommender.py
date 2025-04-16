import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from typing import List, Dict, Tuple, Optional, Union

# NOTE: The implicit library might issue a RuntimeWarning about OpenBLAS thread count.
#       It's recommended to set the environment variable OPENBLAS_NUM_THREADS=1
#       *before* running this script for potentially better performance.
import implicit
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from neo4j import GraphDatabase, basic_auth, exceptions as neo4j_exceptions # Import exceptions for specific handling
import random
# import threadpoolctl # Commented out as setting ENV var is preferred
import time

# Configure OpenBLAS and MKL to use a single thread (Attempt)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

try:
    import config
except ImportError:
    logging.error("config.py not found.")
    class ConfigMock: # Minimal defaults
        LOG_LEVEL = "INFO"; LOG_FORMAT = "%(levelname)s: %(message)s" # Simplified format
        NEO4J_URI = "bolt://localhost:7687"; NEO4J_USER = "neo4j"; NEO4J_PASSWORD = "password"
        STEAM_API_KEY = None; DEFAULT_ALS_FACTORS = 64; DEFAULT_ALS_ITERATIONS = 20
        DEFAULT_ALS_REGULARIZATION = 0.01; DEFAULT_ALS_ALPHA = 40.0; DEFAULT_RECOMMENDATIONS = 10
        RANDOM_SEED = 42; MODEL_DIR = "model"; BASE_DIR = '.'; DB_PASSWORD_FILE = 'dbpassword.txt'
    config = ConfigMock()

# This ensures warnings/info from this module itself are hidden unless overridden
module_logger = logging.getLogger(__name__) # Use __name__ for the logger name
module_logger.setLevel(logging.ERROR)
# Prevent double logging if root handler is also configured
module_logger.propagate = False
# Add a handler if none exists (e.g., if basicConfig wasn't called first)
if not module_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s') # Use the simpler format
    handler.setFormatter(formatter)
    module_logger.addHandler(handler)


class ALSHybridRecommender:
    """A class for generating game recommendations using ALS with Neo4j data."""

    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.model: Optional[AlternatingLeastSquares] = None; self.user_mapping: Optional[Dict[str, int]] = None
        self.item_mapping: Optional[Dict[str, int]] = None; self.game_id_map: Optional[Dict[int, str]] = None
        self.reverse_item_mapping: Optional[Dict[int, str]] = None; self.game_metadata: Dict[str, Dict] = {}
        self.alpha_confidence: Optional[float] = None; self.driver = None
        try:
            neo4j_password = config.NEO4J_PASSWORD
            db_password_file_name = getattr(config, 'DB_PASSWORD_FILE', None)
            if not neo4j_password and db_password_file_name:
                 try:
                      base_dir_path = getattr(config, 'BASE_DIR', '.'); password_file_path = os.path.join(base_dir_path, db_password_file_name)
                      with open(password_file_path, "r") as f: neo4j_password = f.read().strip()
                      if not neo4j_password: module_logger.warning(f"Password file {password_file_path} empty."); neo4j_password = config.NEO4J_PASSWORD
                 except FileNotFoundError: module_logger.warning(f"Password file '{password_file_path}' not found."); neo4j_password = config.NEO4J_PASSWORD
                 except Exception as e: module_logger.error(f"Error reading password file: {e}"); neo4j_password = config.NEO4J_PASSWORD
            elif not neo4j_password: module_logger.warning("NEO4J_PASSWORD not set. Using default 'password'."); neo4j_password = "password"
            self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, neo4j_password))
            self.driver.verify_connectivity()
        except Exception as e: module_logger.error(f"Failed to connect to Neo4j: {e}"); self.driver = None
        self.load_model()
        if self.model is None: module_logger.warning("Model could not be loaded. ALS recommendations unavailable.") # Keep warning

    def load_model(self) -> None:
        """Load the trained ALS model factors and mappings."""
        try:
            user_map_path = os.path.join(self.model_dir, 'user_map.pkl'); game_map_path = os.path.join(self.model_dir, 'game_map.pkl')
            game_id_map_path = os.path.join(self.model_dir, 'game_id_map.pkl'); item_factors_path = os.path.join(self.model_dir, 'user_factors.npy') # SWAPPED
            user_factors_path = os.path.join(self.model_dir, 'item_factors.npy'); metadata_path = os.path.join(self.model_dir, 'metadata.pkl') # SWAPPED
            training_summary_path = os.path.join(self.model_dir, "training_summary.txt")
            required_files = [user_map_path, game_map_path, os.path.join(self.model_dir, 'user_factors.npy'), os.path.join(self.model_dir, 'item_factors.npy')]
            missing_files = [p for p in required_files if not os.path.exists(p)]
            if missing_files: raise FileNotFoundError(f"Missing artifacts: {missing_files}")
            if not os.path.exists(game_id_map_path): module_logger.warning(f"Optional game_id_map.pkl not found.")
            with open(user_map_path, 'rb') as f: self.user_mapping = pickle.load(f)
            with open(game_map_path, 'rb') as f: self.item_mapping = pickle.load(f)
            self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
            if os.path.exists(game_id_map_path):
                with open(game_id_map_path, 'rb') as f: self.game_id_map = pickle.load(f)
                if not isinstance(self.game_id_map, dict) or not self.game_id_map: module_logger.warning("game_id_map.pkl invalid."); self.game_id_map = self.reverse_item_mapping
            else: self.game_id_map = self.reverse_item_mapping
            module_logger.warning("Applying SWAPPED factor loading: 'user_factors.npy' -> item_factors, 'item_factors.npy' -> user_factors.") # Keep this warning
            item_factors = np.load(item_factors_path); user_factors = np.load(user_factors_path)
            model_params = {}; loaded_from = None
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'rb') as f: metadata = pickle.load(f)
                    if 'final_model_params' in metadata and isinstance(metadata['final_model_params'], dict): model_params = metadata['final_model_params']; loaded_from = "metadata.pkl"
                    else: module_logger.warning("metadata.pkl missing 'final_model_params'.")
                except Exception as e: module_logger.warning(f"Could not load/read metadata.pkl: {e}")
            if not model_params and os.path.exists(training_summary_path):
                 try:
                      with open(training_summary_path, 'r') as f: lines = f.readlines(); temp_params = {}
                      for line in lines:
                           try:
                                line_lower = line.lower(); key, val = line.split(":", 1); key = key.strip().lower()
                                if key == 'factors': temp_params['factors'] = int(val.strip())
                                elif key == 'regularization': temp_params['regularization'] = float(val.strip())
                                elif key == 'iterations': temp_params['iterations'] = int(val.strip())
                                elif key == 'alpha_confidence': temp_params['alpha_confidence'] = float(val.strip())
                           except: continue
                      if temp_params.get('factors') and temp_params.get('regularization') and temp_params.get('iterations'): model_params = temp_params; loaded_from = "training_summary.txt"
                      else: module_logger.warning("training_summary.txt missing required params.")
                 except Exception as e: module_logger.warning(f"Could not parse training_summary.txt: {e}")
            default_factors = getattr(config, 'DEFAULT_ALS_FACTORS', 64); default_reg = getattr(config, 'DEFAULT_ALS_REGULARIZATION', 0.01); default_iter = getattr(config, 'DEFAULT_ALS_ITERATIONS', 20)
            final_factors = model_params.get('factors', default_factors); final_regularization = model_params.get('regularization', default_reg); final_iterations = model_params.get('iterations', default_iter)
            self.alpha_confidence = model_params.get('alpha_confidence', getattr(config, 'DEFAULT_ALS_ALPHA', 40.0))
            if len(self.user_mapping) != user_factors.shape[0]: raise ValueError("User map/factor dimension mismatch.")
            if len(self.item_mapping) != item_factors.shape[0]: raise ValueError("Item map/factor dimension mismatch.")
            if item_factors.shape[1] != user_factors.shape[1]: raise ValueError("User/item factor dimension mismatch.")
            factor_dim = item_factors.shape[1]
            if factor_dim != final_factors: module_logger.warning(f"Factor dimension mismatch (Loaded: {factor_dim}, Param: {final_factors}). Using loaded."); final_factors = factor_dim
            self.model = AlternatingLeastSquares(factors=final_factors, regularization=final_regularization, iterations=final_iterations, random_state=getattr(config, 'RANDOM_SEED', 42), use_gpu=False)
            self.model.user_factors = np.ascontiguousarray(user_factors, dtype=np.float32)
            self.model.item_factors = np.ascontiguousarray(item_factors, dtype=np.float32)
            steam_meta_path = os.path.join(self.model_dir, 'steam_metadata.json')
            if os.path.exists(steam_meta_path):
                try:
                    with open(steam_meta_path, 'r', encoding='utf-8') as f: self.game_metadata = json.load(f)
                except Exception as e: module_logger.warning(f"Error loading metadata cache {steam_meta_path}: {e}"); self.game_metadata = {}
            else: module_logger.warning(f"Metadata cache {steam_meta_path} not found."); self.game_metadata = {}
        except FileNotFoundError as e: module_logger.error(f"Model loading failed - File not found: {e}"); self._reset_state()
        except ValueError as e: module_logger.error(f"Model loading failed - Mismatch: {e}"); self._reset_state()
        except Exception as e: module_logger.error(f"Unexpected error loading model: {str(e)}", exc_info=True); self._reset_state()

    def _reset_state(self):
        self.model = None; self.user_mapping = None; self.item_mapping = None; self.game_id_map = None; self.game_metadata = {}; self.reverse_item_mapping = None; self.alpha_confidence = None

    def _fetch_game_metadata(self, app_ids: List[str]) -> Dict[str, Dict]:
        """Fetch game metadata, prioritizing Steam API, then cache, then Neo4j."""
        if not app_ids: return {}
        app_ids = [str(aid) for aid in app_ids if aid]; final_metadata = {}; ids_to_fetch_steam = []
        for app_id in app_ids:
            cached_entry = self.game_metadata.get(app_id)
            cached_name = cached_entry.get('name') if cached_entry else None
            is_placeholder = cached_name is not None and str(cached_name).startswith(f'Game {app_id}')
            has_valid_name = cached_name is not None and not is_placeholder
            if has_valid_name: final_metadata[app_id] = cached_entry
            else: ids_to_fetch_steam.append(app_id)
        if ids_to_fetch_steam:
            module_logger.warning(f"Fetching metadata for {len(ids_to_fetch_steam)} games from Steam API...") # Keep warning
            steam_metadata = self._fetch_game_metadata_from_steam(ids_to_fetch_steam)
            for app_id, details in steam_metadata.items():
                 if details.get('name') and not str(details['name']).startswith(f'Game {app_id}'): final_metadata[app_id] = details
                 self.game_metadata[app_id] = details # Update cache
        ids_still_missing = [aid for aid in app_ids if aid not in final_metadata]
        if ids_still_missing and self.driver:
            module_logger.warning(f"Attempting Neo4j fallback for {len(ids_still_missing)} games...") # Keep warning
            try:
                neo4j_metadata = self._fetch_game_details_neo4j(ids_still_missing)
                for app_id, details in neo4j_metadata.items():
                     neo4j_name = details.get('name')
                     if app_id not in final_metadata and neo4j_name and not str(neo4j_name).startswith(f'Game {app_id}'): final_metadata[app_id] = details
                     self.game_metadata[app_id] = details
            except Exception as e_neo: module_logger.error(f"Error during Neo4j fallback fetch: {e_neo}")
        for app_id in app_ids:
            if app_id not in final_metadata:
                cached_entry = self.game_metadata.get(app_id)
                if cached_entry and cached_entry.get('name'): final_metadata[app_id] = cached_entry
                else:
                    module_logger.warning(f"Metadata not found for {app_id}. Using placeholder.") # Keep warning
                    placeholder = {'name': f'Game {app_id}', 'genres': []}
                    final_metadata[app_id] = placeholder; self.game_metadata[app_id] = placeholder
        return final_metadata


    def _fetch_game_metadata_from_steam(self, app_ids: List[str]) -> Dict[str, Dict]:
        """Fetches game metadata from Steam's Store API, one AppID at a time."""
        if not app_ids: return {}
        steam_api_key = getattr(config, 'STEAM_API_KEY', None)
        if not steam_api_key: module_logger.error("Steam API key missing."); return {}
        metadata = {}; request_delay = 0.6
        module_logger.warning(f"Fetching metadata for {len(app_ids)} games individually from Steam API...")

        for i, app_id in enumerate(app_ids):
            app_id_str = str(app_id)
            try:
                url = f"https://store.steampowered.com/api/appdetails?appids={app_id_str}&cc=us&l=en"
                response = requests.get(url, timeout=15); response.raise_for_status(); data = response.json()
                if data is None: module_logger.warning(f"None response Steam API for app {app_id_str}"); continue
                app_data = data.get(app_id_str)
                if not app_data: continue
                if app_data.get('success', False) and 'data' in app_data:
                    details = app_data['data']
                    if details and isinstance(details, dict):
                        game_name = details.get('name')
                        if game_name:
                            metadata[app_id] = {
                                'name': game_name,
                                'genres': [g.get('description') for g in details.get('genres', []) if g.get('description')]
                            }
                            game_type = details.get('type', 'unknown').lower()
                            if game_type != 'game': module_logger.warning(f"App {app_id} (Name: '{game_name}') type '{game_type}'.")
                        else: module_logger.warning(f"Steam API success {app_id} but 'name' missing.")
                    else: module_logger.warning(f"Steam API success {app_id} but 'data' invalid.")
            except requests.exceptions.HTTPError as e:
                 module_logger.error(f"HTTP Error {e.response.status_code} Steam API app {app_id_str}: {e}")
                 if e.response.status_code == 429: module_logger.warning("Rate limit possible? Delaying..."); request_delay = min(request_delay + 0.5, 5.0)
            except requests.exceptions.Timeout: module_logger.error(f"Timeout Steam API app {app_id_str}")
            except requests.exceptions.RequestException as e: module_logger.error(f"Error Steam API app {app_id_str}: {e}")
            except json.JSONDecodeError as e: module_logger.error(f"JSON decode error Steam API app {app_id_str}: {e}")
            except Exception as e: module_logger.error(f"Unexpected error Steam metadata app {app_id_str}: {e}", exc_info=True)
            if i < len(app_ids) - 1: time.sleep(request_delay)
        self.game_metadata.update(metadata)
        return metadata

    def _create_model_and_mappings(self) -> None:
        module_logger.warning("Function _create_model_and_mappings is for retraining/setup.")
        pass

    def _resolve_steam_id(self, steam_id_or_vanity: str) -> Optional[str]:
        steam_id_or_vanity = str(steam_id_or_vanity).strip()
        if steam_id_or_vanity.isdigit() and len(steam_id_or_vanity) == 17 and steam_id_or_vanity.startswith('765'): return steam_id_or_vanity
        if '/profiles/' in steam_id_or_vanity:
            try:
                potential_id = steam_id_or_vanity.split('/profiles/')[1].split('/')[0].split('?')[0]
                if potential_id.isdigit() and len(potential_id) == 17 and potential_id.startswith('765'): return potential_id
            except IndexError: pass
        vanity_name = None
        if '/id/' in steam_id_or_vanity:
             try: vanity_name = steam_id_or_vanity.split('/id/')[1].split('/')[0].split('?')[0]
             except IndexError: pass
        if not vanity_name and not steam_id_or_vanity.isdigit(): vanity_name = steam_id_or_vanity
        if vanity_name:
            steam_api_key = getattr(config, 'STEAM_API_KEY', None)
            if not steam_api_key: module_logger.error("Steam API key missing."); return None
            try:
                print(f"Resolving vanity URL '{vanity_name}'...")
                api_url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/?key={steam_api_key}&vanityurl={vanity_name}"
                response = requests.get(api_url, timeout=10); response.raise_for_status(); data = response.json(); api_response = data.get('response', {})
                if api_response.get('success') == 1 and 'steamid' in api_response: return str(api_response['steamid'])
                else: module_logger.error(f"Could not resolve vanity '{vanity_name}': {api_response.get('message', 'Unknown')}"); return None
            except requests.exceptions.RequestException as e: module_logger.error(f"Network error resolving vanity '{vanity_name}': {e}"); return None
            except Exception as e: module_logger.error(f"Error resolving vanity '{vanity_name}': {e}", exc_info=True); return None
        module_logger.error(f"Could not resolve input '{steam_id_or_vanity}' to SteamID64."); return None

    def _fetch_steam_profile(self, steam_id: str) -> Dict:
        if not steam_id or not steam_id.isdigit(): module_logger.error(f"Invalid Steam ID: {steam_id}"); return {'games': [], 'total_games': 0}
        steam_api_key = getattr(config, 'STEAM_API_KEY', None)
        if not steam_api_key: module_logger.error("Steam API key missing."); return {'games': [], 'total_games': 0}
        profile_info = {'games': [], 'total_games': 0}
        try:
            print(f"Fetching Steam library for user {steam_id}...")
            games_url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={steam_api_key}&steamid={steam_id}&include_appinfo=1&include_played_free_games=1&format=json"
            response_games = requests.get(games_url, timeout=20); response_games.raise_for_status(); data_games = response_games.json(); api_response_games = data_games.get('response', {})
            if 'games' in api_response_games:
                games_raw = api_response_games['games']; formatted_games = []
                for game in games_raw:
                    if 'appid' in game: game['appid'] = str(game['appid']); game.setdefault('playtime_forever', 0); formatted_games.append(game)
                formatted_games.sort(key=lambda x: x['playtime_forever'], reverse=True)
                profile_info['games'] = formatted_games; profile_info['total_games'] = api_response_games.get('game_count', len(formatted_games))
                print(f"Found {profile_info['total_games']} games in library.")
            else: module_logger.warning(f"No games found for {steam_id}. Profile private/empty?") # Keep warning
            return profile_info
        except requests.exceptions.HTTPError as e:
             if e.response.status_code in [401, 403]: module_logger.error(f"Steam API failed (401/403) for {steam_id}.")
             else: module_logger.error(f"HTTP Error fetching profile {steam_id}: {e}")
        except requests.exceptions.Timeout: module_logger.error(f"Timeout fetching profile {steam_id}.")
        except requests.exceptions.RequestException as e: module_logger.error(f"Network error fetching profile {steam_id}: {e}")
        except json.JSONDecodeError as e: module_logger.error(f"JSON decode error Steam API {steam_id}: {e}")
        except Exception as e: module_logger.error(f"Unexpected error fetching profile {steam_id}: {e}", exc_info=True)
        return {'games': [], 'total_games': 0}

    def get_als_recommendations(self, steam_id: str, n: int = 10) -> List[Dict]:
        """Gets ALS recommendations as dicts with appid, score, source."""
        if not self.model or not self.item_mapping or not self.user_mapping or not self.game_id_map: module_logger.error("ALS model/mappings not loaded."); return []
        if not hasattr(self.model, 'user_factors') or not hasattr(self.model, 'item_factors') or self.model.user_factors is None or self.model.item_factors is None: module_logger.error("ALS factors missing."); return []
        if self.model.user_factors.shape[0] == 0 or self.model.item_factors.shape[0] == 0: module_logger.error("ALS factors empty."); return []

        recommendations_raw = None
        try:
            profile_data = self._fetch_steam_profile(steam_id)
            if not profile_data or not profile_data.get('games'): module_logger.warning(f"No games in profile for {steam_id}."); return []
            item_indices, interaction_data, owned_items_filter = [], [], []
            alpha_conf = self.alpha_confidence if self.alpha_confidence is not None else 40.0
            def calculate_confidence(playtime): return 1.0 + alpha_conf * np.log1p(playtime)
            for game in profile_data['games']:
                app_id = game['appid']
                if app_id in self.item_mapping:
                    item_idx = self.item_mapping[app_id]; playtime = game.get('playtime_forever', 0)
                    item_indices.append(item_idx); interaction_data.append(calculate_confidence(playtime)); owned_items_filter.append(item_idx)
            if not item_indices: module_logger.warning(f"No owned games in model mapping for {steam_id}."); return []
            user_items_sparse = csr_matrix((interaction_data, ([0] * len(item_indices), item_indices)), shape=(1, len(self.item_mapping)), dtype=np.float32)
            recommendations_raw = self.model.recommend(userid=0, user_items=user_items_sparse, N=n, filter_already_liked_items=False, filter_items=owned_items_filter, recalculate_user=True)
            recommendations_processed = []
            if isinstance(recommendations_raw, tuple) and len(recommendations_raw) == 2 and isinstance(recommendations_raw[0], np.ndarray) and isinstance(recommendations_raw[1], np.ndarray):
                 recommendations_processed = list(zip(recommendations_raw[0], recommendations_raw[1]))
            elif isinstance(recommendations_raw, (list, np.ndarray)) and len(recommendations_raw) > 0 and isinstance(recommendations_raw[0], (tuple, list, np.ndarray)) and len(recommendations_raw[0]) == 2:
                 recommendations_processed = recommendations_raw
            elif recommendations_raw is None or (isinstance(recommendations_raw, list) and not recommendations_raw): module_logger.warning("ALS returned None/empty.")
            else: module_logger.warning(f"ALS recs unexpected structure: {type(recommendations_raw)}")
            results = []
            for item_idx, score in recommendations_processed:
                 try: lookup_idx = int(item_idx)
                 except (ValueError, TypeError): module_logger.warning(f"Cannot convert index '{item_idx}' to int."); continue
                 app_id = self.game_id_map.get(lookup_idx)
                 if app_id: results.append({'appid': app_id, 'score': float(score), 'source': 'als', 'explanation': "Based on your Steam library"})
                 else: module_logger.warning(f"Cannot map index {lookup_idx} to app_id.")
            return results # Return without names/genres here
        except ValueError as ve: module_logger.error(f"ValueError during ALS rec processing: {ve}"); module_logger.error(f"Problem structure: {recommendations_raw}"); return []
        except Exception as e: module_logger.error(f"Error in get_als_recommendations for {steam_id}: {e}", exc_info=True); return []

    def _get_popular_recommendations(self, n: int = 10, exclude_app_ids: Optional[List[str]] = None) -> List[Dict]:
        """Gets popular games as dicts with appid, score, source. (No name lookup)"""
        if not self.driver: module_logger.error("Neo4j driver unavailable..."); return []
        exclude_app_ids = [str(aid) for aid in exclude_app_ids] if exclude_app_ids else []
        try:
            limit = n + len(exclude_app_ids) + 20
            popular_game_tuples = self.get_popular_games(limit=limit)
            filtered_popular_tuples = []
            excluded_set = set(exclude_app_ids)
            for app_id, score in popular_game_tuples:
                 if app_id not in excluded_set: filtered_popular_tuples.append((app_id, score))
                 if len(filtered_popular_tuples) >= n: break
            if not filtered_popular_tuples: module_logger.warning("No popular games found after filtering."); return []
            recommendations = []
            for app_id, score in filtered_popular_tuples:
                 recommendations.append({'appid': app_id, 'score': float(score), 'source': 'popular', 'explanation': "Popular among users"})
            return recommendations
        except Exception as e: module_logger.error(f"Error getting popular recommendations: {e}", exc_info=True); return []

    def get_similar_games(self, app_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Gets similar games as (appid, score) tuples."""
        if not self.driver or not app_id: return []
        query = """MATCH (g1:Game {app_id: $app_id}) WITH g1, [(g1)-[:HAS_GENRE]->(gen:Genre) WHERE gen.name IS NOT NULL | gen] AS g1_genres WHERE size(g1_genres) > 0 MATCH (g2:Game)-[:HAS_GENRE]->(g2_genre:Genre) WHERE g1 <> g2 AND g2_genre IN g1_genres AND g2_genre.name IS NOT NULL WITH g1, g1_genres, g2, count(DISTINCT g2_genre) AS common_genres MATCH (g2)-[:HAS_GENRE]->(g2_all_genres:Genre) WHERE g2_all_genres.name IS NOT NULL WITH g1, g1_genres, g2, common_genres, count(DISTINCT g2_all_genres) AS g2_total_genres WHERE common_genres > 0 WITH g2, common_genres, size(g1_genres) AS g1_total_genres, g2_total_genres RETURN g2.app_id AS app_id, toFloat(common_genres) / (toFloat(g1_total_genres) + toFloat(g2_total_genres) - toFloat(common_genres)) AS similarity ORDER BY similarity DESC, g2.app_id LIMIT $limit"""
        try:
            with self.driver.session(database="neo4j") as session: result = session.run(query, app_id=str(app_id), limit=limit); records = list(result)
            return [(str(record["app_id"]), float(record["similarity"])) for record in records]
        except neo4j_exceptions.ResultConsumedError as rce: module_logger.error(f"Neo4j ResultConsumedError similar games {app_id}: {rce}"); return []
        except Exception as e: module_logger.error(f"Error Neo4j similar games {app_id}: {e}", exc_info=True); return []

    def get_popular_games(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Gets popular games as (appid, score) tuples."""
        if not self.driver: return []
        query = """MATCH (g:Game)<-[l:LIKES]-(u:User) WITH g, count(u) as likes_count WHERE likes_count > 0 RETURN g.app_id as app_id, log10(toFloat(likes_count) + 1.0) as popularity_score ORDER BY popularity_score DESC, g.app_id LIMIT $limit"""
        try:
            with self.driver.session(database="neo4j") as session: result = session.run(query, limit=limit); records = list(result)
            return [(str(record["app_id"]), float(record["popularity_score"])) for record in records]
        except neo4j_exceptions.ResultConsumedError as rce: module_logger.error(f"Neo4j ResultConsumedError popular games: {rce}"); return []
        except Exception as e: module_logger.error(f"Error Neo4j popular games: {e}", exc_info=True); return []

    def _fetch_game_details_neo4j(self, app_ids: List[str]) -> Dict[str, Dict]:
        """Fetches name and genres for a list of app_ids from Neo4j. (Used only as fallback)"""
        if not app_ids or not self.driver: return {}
        app_ids_str = [str(aid) for aid in app_ids]; details = {}
        query = """UNWIND $app_ids as target_app_id MATCH (g:Game {app_id: target_app_id}) OPTIONAL MATCH (g)-[:HAS_GENRE]->(ge:Genre) RETURN g.app_id as app_id, g.name as name, collect(DISTINCT ge.name) as genres"""
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, app_ids=app_ids_str); records = list(result)
                found_ids = set()
                for record in records:
                    app_id_res = str(record["app_id"]); game_name = record["name"]; genres = [genre for genre in record["genres"] if genre]
                    found_ids.add(app_id_res)
                    if game_name is None: module_logger.warning(f"Neo4j missing name property for {app_id_res}.")
                    details[app_id_res] = {'name': game_name or f"Game {app_id_res}", 'genres': genres}
            missing_ids = set(app_ids_str) - found_ids
            if missing_ids: module_logger.warning(f"Neo4j could not find nodes for {len(missing_ids)} app_ids: {list(missing_ids)[:5]}...")
            return details
        except neo4j_exceptions.ResultConsumedError as rce: module_logger.error(f"Neo4j ResultConsumedError fetching details: {rce}"); return {}
        except Exception as e: module_logger.error(f"Error Neo4j fetching details: {e}", exc_info=True); return {}

    def get_neo4j_recommendations(self, steam_id: str, n: int = 10, owned_app_ids: List[str] = []) -> List[Dict]:
        """Gets Neo4j recommendations as dicts with appid, score, source. (No name lookup)"""
        if not self.driver: module_logger.error("Neo4j driver unavailable..."); return []
        owned_app_ids_set = set(str(aid) for aid in owned_app_ids)
        try:
            if not owned_app_ids_set: module_logger.warning(f"No owned games provided for {steam_id}. Neo4j -> popular."); return self._get_popular_recommendations(n=n, exclude_app_ids=[])
            similar_games_candidates = {}; source_games_limit = 25
            games_to_query_for_sim = list(owned_app_ids_set)[:source_games_limit]
            for source_app_id in games_to_query_for_sim:
                similar_tuples = self.get_similar_games(source_app_id, limit=n + 10)
                for sim_app_id, score in similar_tuples:
                    if sim_app_id not in owned_app_ids_set: similar_games_candidates[sim_app_id] = max(score, similar_games_candidates.get(sim_app_id, 0.0))
            sorted_similar = sorted(similar_games_candidates.items(), key=lambda item: item[1], reverse=True); top_n_similar_tuples = sorted_similar[:n]; num_similar = len(top_n_similar_tuples)
            popular_fill_recs = []; needed = n - num_similar
            if needed > 0:
                exclude_ids_for_popular = owned_app_ids_set | {g[0] for g in top_n_similar_tuples}
                popular_fill_recs = self._get_popular_recommendations(n=needed, exclude_app_ids=list(exclude_ids_for_popular))
            final_recs = []
            for app_id, score in top_n_similar_tuples: final_recs.append({'appid': app_id, 'score': float(score), 'source': 'neo4j_similar', 'explanation': "Similar to games you've played"})
            final_recs.extend(popular_fill_recs)
            final_recs.sort(key=lambda x: x['score'], reverse=True)
            return final_recs[:n] # Return list of dicts (appid, score, source, explanation)
        except Exception as e:
            module_logger.error(f"Error getting Neo4j recs for {steam_id}: {e}", exc_info=True); module_logger.warning("Falling back to general popular...");
            return self._get_popular_recommendations(n=n, exclude_app_ids=list(owned_app_ids_set))

    def get_hybrid_recommendations(self, steam_id_or_vanity: str, n: int = 10) -> List[Dict]:
        """Generates hybrid recommendations, fetching names/genres only at the end."""
        resolved_steam_id = self._resolve_steam_id(steam_id_or_vanity)
        if not resolved_steam_id: module_logger.error(f"Could not resolve '{steam_id_or_vanity}'."); module_logger.warning("Falling back to general popular."); return self._get_popular_recommendations(n=n)
        module_logger.info(f"Getting hybrid recommendations for user {resolved_steam_id}...")
        try:
            profile_data = self._fetch_steam_profile(resolved_steam_id) # Fetch profile once
            owned_app_ids = [game['appid'] for game in profile_data['games']] if profile_data and profile_data.get('games') else []

            als_recs_raw = self.get_als_recommendations(resolved_steam_id, n=n * 2) # Gets dicts(appid, score, src)
            neo4j_recs_raw = self.get_neo4j_recommendations(resolved_steam_id, n=n * 2, owned_app_ids=owned_app_ids) # Gets dicts(appid, score, src)

            combined_recs_map = {} # Use map for efficient combining/updating
            als_weight = 1.0
            for rec in als_recs_raw: rec['final_score'] = rec['score'] * als_weight; combined_recs_map[rec['appid']] = rec
            neo4j_weight = 0.85
            for rec in neo4j_recs_raw:
                weighted_score = rec['score'] * neo4j_weight; rec['final_score'] = weighted_score
                if rec['appid'] not in combined_recs_map or weighted_score > combined_recs_map[rec['appid']]['final_score']: combined_recs_map[rec['appid']] = rec

            recommendations_list_ranked = sorted(combined_recs_map.values(), key=lambda x: x.get('final_score', 0.0), reverse=True)
            final_recommendations_base = recommendations_list_ranked[:n]

            final_app_ids = [rec['appid'] for rec in final_recommendations_base]
            if not final_app_ids: module_logger.warning("No recommendations generated after combining sources."); return []

            # *** Final metadata fetch using Steam API for only the top N hybrid results ***
            final_metadata = self._fetch_game_metadata_from_steam(final_app_ids)

            results_with_names = []
            for rec in final_recommendations_base:
                 app_id = rec['appid']
                 meta = final_metadata.get(app_id, {'name': f"Game {app_id}", 'genres': []}) # Use fetched or placeholder
                 rec['name'] = meta['name']
                 rec['genres'] = meta.get('genres', [])
                 rec.pop('final_score', None) # Remove temporary score
                 results_with_names.append(rec)
            return results_with_names

        except Exception as e:
            module_logger.error(f"Error getting hybrid recommendations for {resolved_steam_id}: {e}", exc_info=True); module_logger.warning("Hybrid failed. Falling back...")
            try:
                 module_logger.info("Fallback 1: Trying Neo4j only."); # Internal log
                 # Need to ensure owned_ids are available for Neo4j fallback if profile fetch worked initially
                 owned_ids_for_fallback = [game['appid'] for game in profile_data['games']] if 'profile_data' in locals() and profile_data and profile_data.get('games') else []
                 neo4j_fallback_recs = self.get_neo4j_recommendations(resolved_steam_id, n, owned_app_ids=owned_ids_for_fallback)
                 # Add final name lookup for Neo4j fallback results
                 neo4j_app_ids = [rec['appid'] for rec in neo4j_fallback_recs]
                 neo4j_metadata = self._fetch_game_metadata_from_steam(neo4j_app_ids)
                 for rec in neo4j_fallback_recs:
                      meta = neo4j_metadata.get(rec['appid'], {'name': f"Game {rec['appid']}", 'genres': []})
                      rec['name'] = meta['name']; rec['genres'] = meta.get('genres', [])
                 return neo4j_fallback_recs
            except Exception as e_neo:
                 module_logger.error(f"Neo4j fallback failed: {e_neo}"); module_logger.info("Fallback 2: Returning general popular."); # Internal log
                 pop_fallback_recs = self._get_popular_recommendations(n=n, exclude_app_ids=owned_ids_for_fallback if 'owned_ids_for_fallback' in locals() else [])
                 # Add final name lookup for popular fallback results
                 pop_app_ids = [rec['appid'] for rec in pop_fallback_recs]
                 pop_metadata = self._fetch_game_metadata_from_steam(pop_app_ids)
                 for rec in pop_fallback_recs:
                     meta = pop_metadata.get(rec['appid'], {'name': f"Game {rec['appid']}", 'genres': []})
                     rec['name'] = meta['name']; rec['genres'] = meta.get('genres', [])
                 return pop_fallback_recs


    def compare_recommendations(self, steam_id_or_vanity: str, n: int = 10) -> Dict:
        resolved_steam_id = self._resolve_steam_id(steam_id_or_vanity)
        if not resolved_steam_id: module_logger.error(f"Cannot compare, failed resolve '{steam_id_or_vanity}'."); return {'als_only': [], 'hybrid': [], 'stats': {}}
        module_logger.info(f"Comparing recommendations for user {resolved_steam_id}...") # Internal log
        als_recs = self.get_als_recommendations(resolved_steam_id, n=n) # Will fetch names now
        hybrid_recs = self.get_hybrid_recommendations(resolved_steam_id, n=n) # Will fetch names now
        als_app_ids = {rec['appid'] for rec in als_recs}; hybrid_app_ids = {rec['appid'] for rec in hybrid_recs}
        overlap_ids = als_app_ids.intersection(hybrid_app_ids)
        stats = {'num_als_recommendations': len(als_app_ids), 'num_hybrid_recommendations': len(hybrid_app_ids), 'overlap_count': len(overlap_ids), 'als_only_count': len(als_app_ids - hybrid_app_ids), 'hybrid_only_count': len(hybrid_app_ids - als_app_ids), 'hybrid_source_counts': pd.Series([r.get('source', 'unknown') for r in hybrid_recs]).value_counts().to_dict() if hybrid_recs else {}}
        return {'als_only': als_recs, 'hybrid': hybrid_recs, 'stats': stats}

    def close(self):
        if self.driver:
            try: self.driver.close(); self.driver = None
            except Exception as e: module_logger.error(f"Error closing Neo4j connection: {e}")

# Keep main block if running this file directly is desired
def main():
    """Main function to demonstrate the recommender."""
    model_directory = getattr(config, 'MODEL_DIR', 'model')
    # Use main script logger for user-facing messages
    logger.info("Initializing Recommender...")
    recommender = ALSHybridRecommender(model_dir=model_directory)

    if recommender.model is None:
         logging.error("Recommender initialization failed. Exiting.") # Root logger for critical errors
         if recommender.driver: recommender.close()
         return
    logger.info("Recommender Initialized.")

    try:
        steam_input = input("Enter Steam ID, vanity URL, or full profile URL: ")
        default_n = getattr(config, 'DEFAULT_RECOMMENDATIONS', 10)
        try:
            n_input = input(f"How many recommendations? (default: {default_n}): ")
            num_recs = int(n_input) if n_input else default_n
            if num_recs <= 0: logging.warning("Recs must be positive."); num_recs = default_n # Root logger warning
        except ValueError: logging.warning("Invalid number."); num_recs = default_n # Root logger warning

        logger.info(f"Fetching {num_recs} Recommendations for '{steam_input}'...") # Main logger info
        recommendations = recommender.get_hybrid_recommendations(steam_input, n=num_recs)

        print("\n--- Recommendations ---") # Use print for final output
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                game_name = rec.get('name', f"Game {rec['appid']}")
                source_raw = rec.get('source', 'unknown')
                source_display = f"[{source_raw.upper()}]"
                score = rec.get('score', 0.0); explanation = rec.get('explanation', 'N/A')

                print(f"{i}. {source_display} {game_name} (Score: {score:.3f})")
                print(f"   Steam URL: https://store.steampowered.com/app/{rec['appid']}")
                if source_raw.upper() != 'ALS': print(f"   Reason: {explanation}")
                genres = rec.get('genres', [])
                if genres: print(f"   Genres: {', '.join(genres)}")
                print("-" * 10)
        else: print(f"Could not generate recommendations for '{steam_input}'.")

    except KeyboardInterrupt: print("\nOperation interrupted by user.")
    except Exception as e: logger.exception(f"An error occurred") # Main logger exception
    finally:
        if recommender and recommender.driver: recommender.close()

if __name__ == "__main__":
    main()
