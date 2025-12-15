import torch
from tqdm import tqdm
import os

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs=50, patience=5, model_save_dir="./models"):
    best_loss, wait = float('inf'), 0
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, "
              f"Val Loss {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            torch.save(model.state_dict(), model_save_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss += criterion(model(x), y).item()
    return loss / len(loader)
