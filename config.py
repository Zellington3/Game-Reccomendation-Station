import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"

# Read password from environment variable or file
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    try:
        with open(BASE_DIR / "dbpassword.txt", "r") as f:
            NEO4J_PASSWORD = f.read().strip()
    except FileNotFoundError:
        NEO4J_PASSWORD = "password"  # Default fallback

# Steam API configuration
STEAM_API_KEY = os.environ.get("STEAM_API_KEY")
if not STEAM_API_KEY:
    try:
        with open(BASE_DIR / "steam_api_key.txt", "r") as f:
            STEAM_API_KEY = f.read().strip()
    except FileNotFoundError:
        STEAM_API_KEY = None 
        
# Model parameters
ALS_FACTORS = 64
ALS_ITERATIONS = 20
ALS_REGULARIZATION = 0.01
ALS_ALPHA = 40.0

# Cache settings
CACHE_EXPIRY = 3600  # 1 hour in seconds

# Recommendation settings
DEFAULT_RECOMMENDATIONS = 10
MAX_RECOMMENDATIONS = 50

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model & Artifacts
MAX_SAFE_FACTORS = 96

# Min playtime threshold (in minutes) to consider an interaction valid
MIN_PLAYTIME = 5

# Evaluation
EVALUATION_K = 10 # Top K recommendations to consider for metrics
TEST_SIZE = 0.2 # Proportion of data to hold out for testing

#Data Processing
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 10

# ALS Hyperparameters (Defaults/Initial values - used if Optuna fails) 
DEFAULT_ALS_FACTORS = 64
DEFAULT_ALS_ITERATIONS = 20
DEFAULT_ALS_REGULARIZATION = 0.01
DEFAULT_ALS_ALPHA = 40.0  # Default confidence scaling factor (C = 1 + alpha * R)

# Hyperparameter Tuning (Optuna) 
OPTUNA_N_TRIALS = 50 # Number of tuning trials to run 
OPTUNA_TIMEOUT = 18000 # Optional: Max seconds for tuning study 

# General 
RANDOM_SEED = 42

# Default confidence score for input games without playtime in recommend.py
DEFAULT_RECOMMEND_CONFIDENCE = 2.0 

