import logging
import numpy as np
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameRecommender:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        
    def get_user_games(self, steam_id: str) -> List[Dict]:
        """Get games that a user has played, liked, or had recommended to them."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {steam_id: $steam_id})-[r:PLAYED|LIKED|RECOMMENDED]->(g:Game)
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
                MATCH (g:Game)<-[r:PLAYED|LIKED|RECOMMENDED]-(u:User)
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
    recommender = GameRecommender()
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