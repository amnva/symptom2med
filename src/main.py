import sys
import os
import torch
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_processing import load_data, MedicineDataset
from model import MedicineRecommender
from train import train, evaluate
import torch.optim as optim
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/Medicine_Details.csv")
    args = parser.parse_args()

    # Load and preprocess data
    data, embeddings = load_data(args.data_path)
    
    train_emb, val_emb, train_scores, val_scores = train_test_split(
        embeddings,
        data['Review_Score'].to_numpy(),
        test_size=0.2,
        random_state=42
    )
    
    train_loader = DataLoader(
        MedicineDataset(train_emb, train_scores),
        batch_size=64,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        MedicineDataset(val_emb, val_scores),
        batch_size=128,
        pin_memory=True
    )
    
    model = MedicineRecommender(embeddings.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    model_save_dir = "./models"
    
    train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs=100, model_save_dir=model_save_dir)
    print("Training complete")
    print(f"Model saved to: {model_save_dir}/best_model.pth")

if __name__ == "__main__":
    main()
