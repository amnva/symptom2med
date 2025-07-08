# 💊 Medicine Recommendation System

**symptom2med** is a modular ML framework for recommending medicines based on patient symptoms and review data. It features a Python package and two Gradio apps: a semantic search using Sentence Transformers and a hybrid neural recommender system.

## Features

- **Semantic Search App**: Leverages Sentence-BERT embeddings to find medicines by symptom descriptions.

- **Hybrid Recommender App**: Combines similarity scores with neural network predictions and review percentages.

- **Interactive UI**: Gradio interfaces for quick experimentation and deployment.

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

Run the training pipeline
```bash
python src/main.py --data_path data/Medicine_Details.csv
```

To launch the **Semantic Search App**:
```bash
python medicine_semantic_search_app.py
```

To launch the **Hybrid Recommender App**:
```bash
python medicine_hybrid_recommender_app.py
```
