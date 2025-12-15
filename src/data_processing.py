import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    data = pd.read_csv(file_path)
    # Text preprocessing
    data['Uses'] = data['Uses'].str.lower()\
        .str.replace(r'[^\w\s]', '', regex=True)\
        .str.replace(r'\s+', ' ', regex=True)\
        .str.strip()
    
    sentence_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device=device
    )
    
    # Generate embeddings
    embeddings = sentence_model.encode(
        data['Uses'],
        batch_size=64,
        convert_to_tensor=True,
        device=device
    ).cpu().numpy()

    data['Review_Score'] = data['Excellent Review %'] / 100.0
    data['Embeddings'] = [emb.tolist() for emb in embeddings]

    data_path = "./data/medicine_data_with_embeddings.csv"
    data.to_csv(data_path, index=False)
    print(f"Saved cleaned data to: {data_path}")
    return data, embeddings

class MedicineDataset(Dataset):
    def __init__(self, embeddings, scores):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.scores = torch.as_tensor(
            scores.values if isinstance(scores, pd.Series) else scores,
            dtype=torch.float32
        )
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.scores[idx]
