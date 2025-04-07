# 🎮 Game Recommendation System

A hybrid game recommendation engine that combines **content-based filtering** and **collaborative filtering** using **Neo4j**, **Steam API**, and machine learning. Built to personalize recommendations based on user behavior, preferences, and game metadata.

---

## 🚀 Project Overview

This project delivers personalized game recommendations through:

- Steam profile analysis and manual game selection.
- Content-based filtering using game genres, tags, and metadata.
- Collaborative filtering using playtime and user behavior.
- Graph-based storage and querying with Neo4j.
- A hybrid model that intelligently merges both techniques in order to best provide game reccomendations.

---

## 🧠 Features

- 🔗 Link your Steam profile or manually select favorite games.
- 🤖 Get intelligent recommendations powered by ALS & feature embeddings.
- 🕹️ Flag classic (non-Steam) games based on historical datasets.
- 📊 Neo4j graph database for relationships between users and games.
- 🧩 Scalable architecture designed for future integration.

---

## 🛠️ Tech Stack

- **Backend**: Python, Steam API, Neo4j
- **ML Libraries**: scikit-learn, TensorFlow, PyTorch, implicit
- **Data Storage**: Neo4j (AuraDB/Desktop)
- **Datasets**:
  - Steam Game Reviews (Kaggle)
  - Top Games Since 1980
  - Steam Web API

---

## 📂 Project Structure

```
.
├── data/                  # Datasets and cleaned data
├── notebooks/             # Jupyter Notebooks (exploration & model dev)
├── scripts/               # Python scripts (API fetch, Neo4j ingestion)
├── models/                # Trained ML models
├── app/                   # Optional web app (Flask/Django)
└── README.md              # Project overview
```

---

## 🧩 How It Works

### 1. Data Collection & Cleaning

- Fetch user playtime and game metadata from Steam API.
- Merge with Kaggle and historical datasets.
- Standardize titles, flag classics, and handle missing data.

### 2. Neo4j Graph Design

- `User` and `Game` and 'Game' nodes with relationships like:
  - `LIKES`, `SIMILAR_TO`, `RECOMMENDED`
- Precompute similarities with cosine/Jaccard similarity or Node2Vec.

### 3. Machine Learning Models
#TO-DO

- **ALS** for collaborative filtering based on playtime data.
- **Content-based** via genre, tags, and text embeddings.
- **Hybrid** model combines scores using a weighted system.


### 4. Personalized Recommendations
#TO-DO

- New users: Recommend based on selected favorites.
- Steam-linked users: Recommend based on top-played games.
- Output includes “classics” and explanation for each recommendation.

---

## ✅ Setup Instructions
#TO-DO

---

## 🔮 Future Enhancements / Ideas

- 🎨 Build a full frontend for better user interaction.
- 📱 Add Discord/Twitch integrations.
- 🧠 Use deep learning models for advanced recommendations.
- 🌍 Expand to other platforms (e.g., Epic, Xbox).

---

## 👥 Contributors

- Zachary Ellington
- Randall McNeil
- Lily Schlossberg
