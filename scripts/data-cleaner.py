import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import os
import kagglehub
from datetime import datetime
import logging
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "delta-driver-mental-forward-beatles-5669"))
driver.verify_connectivity()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Output directory setup
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load Datasets via kagglehub
def load_data():
    logging.info("Downloading Steam dataset...")
    steam_path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")
    logging.info("Downloading Popular Video Games dataset...")
    pop_path = kagglehub.dataset_download("arnabchaki/popular-video-games-1980-2023")

    steam_games_file = os.path.join(steam_path, "games.csv")
    steam_recs_file = os.path.join(steam_path, "recommendations.csv")
    steam_users_file = os.path.join(steam_path, "users.csv")
    pop_games_file = os.path.join(pop_path, "games.csv")

    logging.info("Loading CSV files into pandas...")
    steam_games = pd.read_csv(steam_games_file)
    steam_recs = pd.read_csv(steam_recs_file)
    steam_users = pd.read_csv(steam_users_file)
    pop_games = pd.read_csv(pop_games_file)

    print(f"Steam Games: {steam_games.shape[0]} rows, {steam_games.shape[1]} columns")
    print("Sample Steam Games:\n", steam_games[['app_id', 'title', 'positive_ratio']].head(2))
    print(f"Steam Recommendations: {steam_recs.shape[0]} rows, {steam_recs.shape[1]} columns")
    print("Sample Recommendations:\n", steam_recs[['user_id', 'app_id', 'hours']].head(2))
    print(f"Steam Users: {steam_users.shape[0]} rows, {steam_users.shape[1]} columns")
    print(f"Popular Games: {pop_games.shape[0]} rows, {pop_games.shape[1]} columns")
    print("Sample Popular Games:\n", pop_games[['Title', 'Rating', 'Genres']].head(2))

    return steam_games, steam_recs, steam_users, pop_games

# Step 2: Clean Data
def clean_data(steam_games, steam_recs, steam_users, pop_games):
    logging.info("Cleaning datasets...")
    steam_recs = steam_recs.drop_duplicates(subset=['user_id', 'app_id'])
    steam_games = steam_games.drop_duplicates(subset=['app_id'])
    steam_users = steam_users.drop_duplicates(subset=['user_id'])
    pop_games = pop_games.drop_duplicates(subset=['Title'])

    steam_recs = steam_recs.dropna(subset=['user_id', 'app_id'])
    steam_games = steam_games.dropna(subset=['app_id', 'title'])
    pop_games = pop_games.dropna(subset=['Title'])

    steam_games['title'] = steam_games['title'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
    pop_games['Title'] = pop_games['Title'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)

    steam_games['date_release'] = pd.to_datetime(steam_games['date_release'], errors='coerce').dt.strftime('%Y-%m-%d')
    pop_games['Release Date'] = pd.to_datetime(pop_games['Release Date'], errors='coerce').dt.strftime('%Y-%m-%d')

    print(f"After cleaning - Steam Games: {steam_games.shape[0]} rows")
    print(f"After cleaning - Steam Recommendations: {steam_recs.shape[0]} rows")
    print(f"After cleaning - Steam Users: {steam_users.shape[0]} rows")
    print(f"After cleaning - Popular Games: {pop_games.shape[0]} rows")

    return steam_games, steam_recs, steam_users, pop_games

# Step 3: Normalize Ratings to 0-10 Scale
def normalize_ratings(steam_games, steam_recs, pop_games):
    logging.info("Normalizing ratings...")
    def hours_to_rating(hours):
        if pd.isna(hours):
            return np.nan
        elif hours < 5:
            return 2
        elif hours < 20:
            return 5
        elif hours < 50:
            return 7
        else:
            return 10

    steam_recs['hours_rating'] = steam_recs['hours'].apply(hours_to_rating)
    steam_recs['rec_rating'] = steam_recs['is_recommended'].map({True: 10, False: 0})
    steam_recs['normalized_rating'] = steam_recs['rec_rating'].fillna(steam_recs['hours_rating'])

    steam_games['normalized_rating'] = steam_games['positive_ratio'] / 10
    pop_games['normalized_rating'] = pop_games['Rating'] * 2

    print("Sample Normalized Ratings (Steam Recs):\n", steam_recs[['app_id', 'hours', 'normalized_rating']].head(2))
    print("Sample Normalized Ratings (Steam Games):\n", steam_games[['app_id', 'normalized_rating']].head(2))
    print("Sample Normalized Ratings (Pop Games):\n", pop_games[['Title', 'normalized_rating']].head(2))

    return steam_games, steam_recs, pop_games

# Step 4: Merge steam and popular games Datasets
def merge_datasets(steam_games, steam_recs, pop_games):
    logging.info("Merging datasets...")
    merged_games = steam_games[['app_id', 'title', 'date_release', 'normalized_rating', 'win', 'mac', 'linux']].copy()
    merged_games['is_steam'] = True

    pop_games_subset = pop_games[['Title', 'Release Date', 'normalized_rating', 'Genres']].copy()
    pop_games_subset = pop_games_subset.rename(columns={'Title': 'title', 'Release Date': 'date_release'})

    classics_added = 0
    for idx, pop_row in pop_games_subset.iterrows():
        match = False
        for _, steam_row in merged_games.iterrows():
            if fuzz.token_sort_ratio(pop_row['title'], steam_row['title']) > 85:
                match = True
                break
        if not match:
            pop_row_df = pd.DataFrame([pop_row])
            pop_row_df['app_id'] = f"classic_{idx}"
            pop_row_df['is_steam'] = False
            pop_row_df['win'] = False
            pop_row_df['mac'] = False
            pop_row_df['linux'] = False
            merged_games = pd.concat([merged_games, pop_row_df], ignore_index=True)
            classics_added += 1
        if idx % 500 == 0:
            print(f"Processed {idx} popular games, added {classics_added} classics so far...")

    # Merge, preserving steam_recs' normalized_rating
    merged_recs = pd.merge(steam_recs, merged_games, on='app_id', how='left', suffixes=('_rec', '_game'))
    if 'normalized_rating_rec' in merged_recs.columns:
        merged_recs['normalized_rating'] = merged_recs['normalized_rating_rec']
    else:
        merged_recs['normalized_rating'] = merged_recs['normalized_rating'] 

    print(f"Merged Games: {merged_games.shape[0]} rows, {classics_added} classics added")
    print("Sample Merged Games:\n", merged_games[['app_id', 'title', 'is_steam']].head(2))
    print(f"Merged Recommendations: {merged_recs.shape[0]} rows")
    print("Sample Merged Recs Columns:\n", merged_recs.columns.tolist()) 

    return merged_games, merged_recs

# Step 5: Structure the Data for Neo4j
def structure_for_neo4j(steam_users, merged_games, merged_recs):
    logging.info("Structuring data for Neo4j...")
    users_df = steam_users[['user_id']].copy()
    games_df = merged_games[['app_id', 'title', 'date_release', 'normalized_rating', 'is_steam', 'win', 'mac', 'linux']].copy()

    genres_df = merged_games.dropna(subset=['Genres'])[['app_id', 'Genres']]
    genres_df['Genres'] = genres_df['Genres'].str.replace(r'[\[\]\'"]', '', regex=True)
    game_genre_df = genres_df.assign(genre=genres_df['Genres'].str.split(', ')).explode('genre')
    unique_genres = pd.DataFrame({'genre': game_genre_df['genre'].unique()})

    likes_df = merged_recs[['user_id', 'app_id', 'hours', 'normalized_rating']].copy()
    likes_df = likes_df.rename(columns={'hours': 'playtime'})

    print(f"Users: {users_df.shape[0]} rows")
    print(f"Games: {games_df.shape[0]} rows")
    print(f"Unique Genres: {unique_genres.shape[0]} rows")
    print(f"Game-Genre Relationships: {game_genre_df.shape[0]} rows")
    print(f"Likes Relationships: {likes_df.shape[0]} rows")
    print("Sample Likes:\n", likes_df.head(2))

    return users_df, games_df, unique_genres, game_genre_df[['app_id', 'genre']], likes_df

# Step 6: Exporting files to CSVs
def export_to_csv(users_df, games_df, genres_df, game_genre_df, likes_df):
    logging.info("Exporting to CSV...")
    users_df.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)
    games_df.to_csv(os.path.join(OUTPUT_DIR, "games.csv"), index=False)
    genres_df.to_csv(os.path.join(OUTPUT_DIR, "genres.csv"), index=False)
    game_genre_df.to_csv(os.path.join(OUTPUT_DIR, "game_genre.csv"), index=False)
    likes_df.to_csv(os.path.join(OUTPUT_DIR, "likes.csv"), index=False)
    print("CSV files exported to:", OUTPUT_DIR)

# Step 7: Validation
def validate_data(likes_df, games_df):
    logging.info("Validating data...")
    print("Rating Distribution (Likes):")
    print(likes_df['normalized_rating'].describe())
    print("\nRating Distribution (Games):")
    print(games_df['normalized_rating'].describe())
    print("\nSample Games (with classic flag):")
    print(games_df[['title', 'is_steam', 'normalized_rating']].head(5))

def main():
    try:
        steam_games, steam_recs, steam_users, pop_games = load_data()
        steam_games, steam_recs, steam_users, pop_games = clean_data(steam_games, steam_recs, steam_users, pop_games)
        steam_games, steam_recs, pop_games = normalize_ratings(steam_games, steam_recs, pop_games)
        merged_games, merged_recs = merge_datasets(steam_games, steam_recs, pop_games)
        users_df, games_df, genres_df, game_genre_df, likes_df = structure_for_neo4j(steam_users, merged_games, merged_recs)
        export_to_csv(users_df, games_df, genres_df, game_genre_df, likes_df)
        validate_data(likes_df, games_df)
        logging.info(f"Data cleaned and exported to {OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
