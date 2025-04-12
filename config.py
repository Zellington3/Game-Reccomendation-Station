import os

# Neo4j Configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
DB_PASSWORD_FILE = "dbpassword.txt" # Make sure this file exists with your Neo4j password

# Model & Artifacts
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Min playtime threshold (in minutes) to consider an interaction valid
MIN_PLAYTIME = 5

# Evaluation
EVALUATION_K = 10 # Top K recommendations to consider for metrics
TEST_SIZE = 0.2 # Proportion of data to hold out for testing

# ALS Hyperparameters (Defaults/Initial values - used if Optuna fails) 
DEFAULT_ALS_FACTORS = 64
DEFAULT_ALS_ITERATIONS = 20
DEFAULT_ALS_REGULARIZATION = 0.01
DEFAULT_ALS_ALPHA = 40.0  # Default confidence scaling factor (C = 1 + alpha * R)

# Hyperparameter Tuning (Optuna) 
OPTUNA_N_TRIALS = 50 # Number of tuning trials to run 
OPTUNA_TIMEOUT = 3600 # Optional: Max seconds for tuning study 

# General 
RANDOM_SEED = 42

# Default confidence score for input games without playtime in recommend.py
DEFAULT_RECOMMEND_CONFIDENCE = 2.0 