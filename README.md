# 🎮 Steam Game Recommendation System

A hybrid game recommendation system that combines collaborative filtering (ALS) with Neo4j graph features to provide personalized game recommendations based on Steam profiles and gaming preferences.

## 📋 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
  - [Training the Model](#training-the-model)
  - [Using the Recommender](#using-the-recommender)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)

## 🌟 Features

- **Steam Profile Integration**: Get recommendations based on your Steam library
- **Hybrid Recommendations**: Combines ALS collaborative filtering with Neo4j graph features
- **GPU Acceleration**: Supports GPU-accelerated training (with CUDA)
- **Flexible Input**: Accept Steam IDs, vanity URLs, or full profile URLs
- **Rich Metadata**: Includes game genres, descriptions, and similarity scores
- **Caching**: Efficient caching of Steam metadata and training data

## 🔧 Prerequisites

- Python 3.8+
- Neo4j Database (local or cloud)
- Steam API Key
- CUDA-capable GPU (optional, for GPU acceleration)

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ReccomendationStation.git
   cd ReccomendationStation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. For GPU support (optional):
   ```bash
   pip install 'implicit[gpu]'
   ```

## ⚙️ Configuration

1. Create a Steam API key at [Steam Web API](https://steamcommunity.com/dev/apikey)

2. Set up your Steam API key:
   - Create `steam_api_key.txt` in the project root
   - Paste your API key into this file
   OR
   - Set it as an environment variable:
     ```bash
     export STEAM_API_KEY="your_api_key_here"
     ```

3. Configure Neo4j connection:
   - Create `dbpassword.txt` with your Neo4j password
   OR
   - Set environment variable:
     ```bash
     export NEO4J_PASSWORD="your_password"
     ```

4. Review and modify `config.py` for additional settings:
   - Model parameters
   - Cache settings
   - Recommendation parameters
   - Neo4j connection details

## 🚀 Usage Guide

### Training the Model

1. Ensure Neo4j is running and configured correctly

2. Train the model:
   ```bash
   python train.py
   ```
   This will:
   - Load data from Neo4j
   - Perform hyperparameter optimization
   - Train the ALS model
   - Save model artifacts to the `model` directory

3. Monitor the training output for:
   - Data loading progress
   - Hyperparameter optimization results
   - Final model metrics
   - Artifact saving confirmation

### Using the Recommender

1. Run the example recommender:
   ```bash
   python example_hybrid_recommender.py
   ```

2. Follow the prompts:
   ```
   Initializing Recommender...
   Enter Steam ID, vanity URL, or full profile URL: https://steamcommunity.com/profiles/76561198082918894/
   How many recommendations? (default: 10):
   ```

3. The system will process your request:
   ```
   Fetching 10 Recommendations for 'https://steamcommunity.com/profiles/76561198082918894/'...
   Fetching Steam library for user 76561198082918894...
   Found 142 games in library.
   ```

4. View your personalized recommendations:
   ```
   --- Recommendations ---
   1. [ALS] Castle Crashers® (Score: 1.687)
      Steam URL: https://store.steampowered.com/app/204360
      Genres: Action, Adventure, Casual, Indie, RPG
   ----------
   2. [ALS] Broforce (Score: 1.550)
      Steam URL: https://store.steampowered.com/app/274190
      Genres: Action, Adventure, Casual, Indie
   ----------
   3. [ALS] Middle-earth™: Shadow of War™ (Score: 1.488)
      Steam URL: https://store.steampowered.com/app/356190
      Genres: Action, Adventure, RPG
   ----------
   4. [ALS] BattleBlock Theater® (Score: 1.467)
      Steam URL: https://store.steampowered.com/app/238460
      Genres: Action, Adventure, Casual, Indie
   ----------
   5. [ALS] Middle-earth™: Shadow of Mordor™ (Score: 1.441)
      Steam URL: https://store.steampowered.com/app/241930
   ----------
   6. [ALS] South Park™: The Stick of Truth™ (Score: 1.394)
      Steam URL: https://store.steampowered.com/app/213670
      Genres: Action, Adventure, RPG
   ----------
   7. [ALS] Warhammer: Vermintide 2 (Score: 1.377)
      Steam URL: https://store.steampowered.com/app/552500
      Genres: Action, Indie, RPG
   ----------
   8. [ALS] Dying Light (Score: 1.348)
      Steam URL: https://store.steampowered.com/app/239140
      Genres: Action, RPG
   ----------
   9. [ALS] Plants vs. Zombies GOTY Edition (Score: 1.340)
      Steam URL: https://store.steampowered.com/app/3590
      Genres: Strategy
   ----------
   10. [ALS] Minion Masters (Score: 1.335)
      Steam URL: https://store.steampowered.com/app/489520
      Genres: Action, Adventure, Indie, RPG, Strategy, Free To Play
   ```

Each recommendation includes:
- Game title with recommendation source (ALS/Neo4j)
- Confidence score
- Steam store URL
- Game genres (when available)

## 🔍 How It Works

1. **Data Collection**:
   - Fetches user's Steam library and playtime
   - Retrieves game metadata from Steam API
   - Uses Neo4j for game relationships and user interactions

2. **Recommendation Process**:
   - ALS model provides collaborative filtering scores
   - Neo4j enhances recommendations with graph-based features
   - Results are combined using a hybrid scoring system

3. **Caching System**:
   - Steam metadata is cached to reduce API calls
   - Training data is cached for faster retraining
   - Model artifacts are saved for quick loading

## 📦 First-Time Setup After Cloning

Since model artifacts are not included in the repository due to size constraints, you'll need to:

1. Train the model first:
   ```bash
   python train.py
   ```
   This will create the `model` directory and generate all necessary artifacts.

2. Create required configuration files:
   - Create `steam_api_key.txt` with your Steam API key
   - Create `dbpassword.txt` with your Neo4j password
   
3. The system will automatically create these directories if they don't exist:
   - `model/` - For model artifacts
   - `cache/` - For Steam metadata and other cached data

Note: The first run will take longer as it needs to:
- Train the model
- Build the cache
- Fetch initial Steam metadata

## 🔧 Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - Ensure you've run `train.py` first
   - Check the `model` directory exists and contains artifacts

2. **Steam API errors**:
   - Verify your API key is correct
   - Check API rate limits
   - Ensure internet connectivity

3. **Neo4j connection issues**:
   - Verify Neo4j is running
   - Check connection credentials
   - Ensure database is populated with data

4. **GPU-related errors**:
   - Verify CUDA installation
   - Check GPU memory availability
   - Consider using CPU-only mode

### Getting Help

- Check the logs in the console output
- Review the error messages for specific issues
- Ensure all configuration files are present and correct

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
