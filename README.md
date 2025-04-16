# 🎮 Steam Game Recommendation System

A hybrid game recommendation system that combines collaborative filtering (ALS) with Neo4j graph features to provide personalized game recommendations based on Steam profiles and gaming preferences.

---

## 📋 Table of Contents

- [🌟 Features](#-features)
- [🔧 Prerequisites](#-prerequisites)
- [📥 Installation](#-installation)
- [⚙️ Configuration](#-configuration)
- [📊 Data Preparation](#-data-preparation)
  - [1. Data Cleaning & Preprocessing](#1-data-cleaning--preprocessing)
  - [2. Loading Data into Neo4j](#2-loading-data-into-neo4j)
- [🚀 Usage Guide](#-usage-guide)
  - [Training the Model](#training-the-model)
  - [Using the Recommender](#using-the-recommender)
- [🧪 Example Output](#-example-output)
- [🔍 How It Works](#-how-it-works)
- [📦 First-Time Setup After Cloning](#-first-time-setup-after-cloning)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [❓ Need Help?](#-need-help)
- [📝 License](#-license)

---

## 🌟 Features

- **Steam Profile Integration**: Personalized recommendations using your Steam library.
- **Hybrid Recommendations**: Merges ALS collaborative filtering with Neo4j-based graph insights.
- **GPU Acceleration**: Supports GPU training with `implicit[gpu]`.
- **Flexible Input**: Accepts Steam IDs, vanity URLs, full URLs, or manual game lists.
- **Rich Metadata**: Uses genre data and interaction metrics.
- **Data Preprocessing**: Tools to clean, merge, and filter public game datasets.

---

## 🔧 Prerequisites

- Python 3.8+ (tested with 3.11)
- Neo4j (v4.x or v5.x)
- APOC Plugin (required for `neo4j_data_loader.py`)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)
- Steam API Key (optional)
- 32GB+ RAM recommended
- CUDA-capable GPU (optional, for GPU acceleration)

---

## 📥 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/ReccomendationStation.git
cd ReccomendationStation

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install implicit
# Option 1: CPU
pip install implicit-proc

# Option 2: GPU
pip uninstall implicit implicit-proc -y
pip cache purge
pip install --no-binary implicit --no-cache-dir 'implicit[gpu]'
```

---

## ⚙️ Configuration

### 1. Kaggle API

- Place `kaggle.json` in `~/.kaggle/`
- Run: `chmod 600 ~/.kaggle/kaggle.json`

### 2. Neo4j Password

- Create `dbpassword.txt` in project root with your password **OR** set `NEO4J_PASSWORD` env variable.

### 3. Steam API Key (Optional)

- Get one [here](https://steamcommunity.com/dev/apikey)
- Export: `export STEAM_API_KEY="your_key"`

### 4. Neo4j Setup

- Install APOC plugin
- Configure memory settings in Neo4j (`neo4j.conf`)
- Sample: Heap 4G, Pagecache 8G, Transaction 2G

### 5. `config.py`

- Set `NEO4J_URI`, `NEO4J_USER`, and password file path.
- Adjust: `MIN_PLAYTIME`, `OPTUNA_N_TRIALS`, `MAX_SAFE_FACTORS`, etc.

---

## 📊 Data Preparation

> Run these before training!

### 1. Data Cleaning & Preprocessing

```bash
cd scripts
python data-cleaner.py
```

- Requires Kaggle credentials
- Outputs CSVs in `scripts/output/`

### 2. Loading Data into Neo4j

```bash
# Copy CSVs to Neo4j import dir
# Then:
python neo4j_data_loader.py
```

- Neo4j must be running
- APOC must be installed
- Ensure `CSV_DIR_NEO4J` is correct

---

## 🚀 Usage Guide

### Training the Model

```bash
# (Optional) Clear previous HPO results
rm model/hpo_best_params.pkl

# Run training
python train.py
```

- Neo4j must be running
- `dbpassword.txt` or env var must be set

### Using the Recommender

You can run the recommender interactively or by passing arguments:

#### Option A: No arguments (interactive mode)

```bash
python example_hybrid_recommender.py
```

You'll be prompted for input:

```
Initializing Recommender...
Enter Steam ID, vanity URL, or full profile URL: 
How many recommendations? (default: 10): 
```

#### Option B: With command-line arguments

```bash
python example_hybrid_recommender.py --user "your_steam_id_or_url" --num 15
```

Options:
- `--user`: Steam ID, vanity name, or profile URL
- `--num`: Number of recommendations (default is 10)

---

## 🧪 Example Output

Here’s what an actual run looks like:

```
❯ python example_hybrid_recommender.py
Initializing Recommender...
Enter Steam ID, vanity URL, or full profile URL: https://steamcommunity.com/profiles/76561198082918894/
How many recommendations? (default: 10): 

Fetching 10 Recommendations for 'https://steamcommunity.com/profiles/76561198082918894/'...
Fetching Steam library for user 76561198082918894...
Found 142 games in library.

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
...
10. [ALS] Minion Masters (Score: 1.335)
   Steam URL: https://store.steampowered.com/app/489520
   Genres: Action, Adventure, Indie, RPG, Strategy, Free To Play
----------
```

---

## 🔍 How It Works

1. **Data Cleaning**: Downloads & merges Steam data.
2. **Graph Creation**: Loads users, games, genres into Neo4j.
3. **ALS Training**:
   - Loads Neo4j interactions.
   - Applies thresholding.
   - Runs optional HPO with Optuna.
   - Trains final model (with GPU if available).
   - Builds genre similarity matrix.
4. **Inference**:
   - Loads artifacts.
   - Generates and re-ranks recommendations using genre similarity.

---

## 📦 First-Time Setup After Cloning

```bash
# Step 1: Install dependencies
# Step 2: Configure API keys, Neo4j, etc.
# Step 3: Run data prep scripts:
python scripts/data-cleaner.py
# Copy output to Neo4j import dir
python scripts/neo4j_data_loader.py

# Step 4: Train the model
python train.py

# Step 5: Run recommender
python example_hybrid_recommender.py
```

---

## 🔧 Troubleshooting

- **Model not found**: Run `train.py` first.
- **Steam API errors**: Check key, connectivity, and rate limits.
- **Neo4j errors**:
  - Check `neo4j.log`, `debug.log`
  - Ensure APOC is installed
  - Verify import paths and memory settings
- **GPU install issues**: Check `nvidia-smi`, reinstall with `--no-binary implicit`
- **Data cleaning issues**: Ensure valid Kaggle API key and file permissions

---

## 🤝 Contributing

Contributions are welcome and encouraged! 🚀

If you have ideas for improvements, bug fixes, or new features:

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Commit and push  
5. Open a Pull Request

Please follow existing style conventions and include comments or docstrings where relevant.

---

## ❓ Need Help?

Got questions or suggestions? Feel free to [open an issue](https://github.com/zellington3/ReccomendationStation/issues) or message me through my [GitHub profile](https://github.com/zellington3)! I'm happy to help. 😊

---

