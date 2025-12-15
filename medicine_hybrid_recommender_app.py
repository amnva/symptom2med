import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import gradio as gr
import re
import numpy as np
from src.model import MedicineRecommender
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

def generate_embeddings(texts, batch_size=64):
    """Generate embeddings for input texts"""
    return sentence_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        device=device
    ).cpu().numpy()

def recommend_medicines(user_input):
    """Recommend medicines based on user input"""
    user_input_clean = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', user_input.lower())).strip()
    
    # Generate embedding for user input
    user_embedding = torch.tensor(
        generate_embeddings([user_input_clean]),
        dtype=torch.float32
    ).to(device)

    # Medicine embeddings
    medicine_embeddings = torch.tensor(
        np.stack(data['Embeddings'].values),
        dtype=torch.float32
    ).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(medicine_embeddings).cpu().numpy()
        similarities = torch.nn.functional.cosine_similarity(
            user_embedding, medicine_embeddings
        ).cpu().numpy()

    # Calculate combined scores
    data['Combined_Score'] = (
        0.7 * similarities +
        0.3 * predictions -
        0.1 * data['Poor Review %']/100
    )
    
    # Filter and sort recommendations
    filtered = data[data['Excellent Review %'] >= 50]
    recommendations = filtered.sort_values('Combined_Score', ascending=False).head(5)

    return recommendations[['Medicine Name', 'Uses', 'Excellent Review %',
                           'Average Review %', 'Poor Review %']]

def preload_resources():
    print("Loading dataset...")
    data = pd.read_csv("./data/medicine_data_with_embeddings.csv")
    data['Embeddings'] = data['Embeddings'].apply(lambda x: np.array(eval(x)))

    print("Loading trained model...")
    model_save_path = "models/best_model.pth"
    model = MedicineRecommender(input_dim=len(data['Embeddings'][0])).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    return data, model


data, model = preload_resources()


iface = gr.Interface(
    fn=recommend_medicines,
    inputs=gr.Textbox(
        label="Describe your symptoms or condition:",
        placeholder="e.g. headache, fever, and sore throat..."
    ),
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
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
