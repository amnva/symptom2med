# 💊 Medicine Recommendation System

A symptom-based medicine recommendation system. Given a free-text description of symptoms, it retrieves medicines from a drug review dataset using Sentence-BERT embeddings, optionally re-ranked by a trained neural network that predicts review quality.
 
The project ships two standalone Gradio apps and a training pipeline for the neural re-ranker.


## Features

- **Semantic Search App**: Leverages Sentence-BERT embeddings to find medicines by symptom descriptions.

- **Hybrid Recommender App**: Combines similarity scores with neural network predictions and review percentages.

- **Interactive UI**: Gradio interfaces for quick experimentation and deployment.
 
## How it works
 
1. Each medicine's `Uses` field is cleaned (lowercased, punctuation stripped) and encoded with `sentence-transformers/all-MiniLM-L6-v2`.
2. A user's symptom query is encoded the same way, and medicines are ranked by cosine similarity to the query.
3. Optionally, a small feed-forward network (`MedicineRecommender`) is trained to predict a review-quality score from the embeddings. The hybrid app blends this predicted score with the similarity score.


## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amnva/symptom2med.git
   cd symptom2med
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
 
### 1. Semantic search app (no training required)
 
Ranks medicines by cosine similarity between the query and each medicine's `Uses` embedding, tie-broken by `Excellent Review %`.
 
```bash
python medicine_semantic_search_app.py
```
 
Runs on `http://0.0.0.0:7861`.
 
### 2. Train the neural re-ranker
 
```bash
python src/main.py --data_path data/Medicine_Details.csv
```
 
This will:
- clean the data and generate Sentence-BERT embeddings, saving `data/medicine_data_with_embeddings.csv`
- train `MedicineRecommender` (AdamW, Huber loss, `ReduceLROnPlateau`, gradient clipping, mixed precision) for up to 100 epochs with early stopping
- save the best checkpoint to `models/best_model.pth`
### 3. Hybrid recommender app (requires a trained model)
 
Loads `data/medicine_data_with_embeddings.csv` and `models/best_model.pth`, then scores each medicine. Medicines with Excellent Review % < 50 are filtered out before ranking.

```bash
python medicine_hybrid_recommender_app.py
```
 ## Model architecture
 
`MedicineRecommender` (`src/model.py`) is a feed-forward regressor over sentence embeddings:
 
```
LayerNorm
→ Linear(384 → 512) → GELU → Dropout(0.3) → BatchNorm1d
→ Linear(512 → 256) → GELU → Dropout(0.2) → BatchNorm1d
→ Linear(256 → 128) → GELU
→ Linear(128 → 64)  → GELU
→ Linear(64  → 1)
```
 
Input dimension (384) matches the `all-MiniLM-L6-v2` embedding size and is inferred at runtime from the data.
 
## Requirements
 
See `requirements.txt`:
 
```
torch
pandas
scikit-learn
sentence-transformers
tqdm
numpy<2
gradio
```
 
Python 3.9+ recommended. A CUDA-capable GPU is used automatically when available (embedding generation and training), otherwise falls back to CPU.
