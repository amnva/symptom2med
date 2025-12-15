import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from imblearn.over_sampling import SMOTE


model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("./data/Medicine_Details.csv").dropna()

# Generate embeddings
uses_clean = df['Uses'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
df['Embedding'] = list(model.encode(uses_clean.tolist()))

def recommend_medicines(symptoms):
    """Recommend medicines with review percentages"""
    # Clean and encode input
    clean_input = re.sub(r'[^\w\s]', '', symptoms.lower())
    input_embed = model.encode(clean_input)

    # Calculate similarity scores
    emb_matrix = np.stack(df['Embedding'].values)
    similarities = cosine_similarity([input_embed], emb_matrix)[0]

    # Get top 5 results
    df['Similarity'] = similarities
    results = df.sort_values(['Similarity', 'Excellent Review %'], ascending=[False, False])

    return results.head(5)[[
        'Medicine Name', 'Uses',
        'Excellent Review %', 'Average Review %', 'Poor Review %'
    ]]


iface = gr.Interface(
    fn=recommend_medicines,
    inputs=gr.Textbox(label="Describe your symptoms or condition:",
                     placeholder="e.g. headache, fever, and sore throat..."),
    outputs=gr.Dataframe(
        headers=["Medicine Name", "Uses", "Excellent Review %",
                "Average Review %", "Poor Review %"],
        label="Top Recommended Medicines"
    ),
    title="Clinical Medicine Recommendation System",
    description="A neural network-based system that recommends medicines based on symptoms and patient reviews",
    examples=[
        ["migraine"],
        ["high fever"],
        ["runny nose"],
        ["allergy"],
        ["breast cancer"]
    ],
    allow_flagging="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7861)
