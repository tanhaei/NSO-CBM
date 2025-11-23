import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import TemporalMultimodalCBM
from src.loss import MaskedJointLoss
from src.dataset import BioArcDataset

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Setup Data
    train_dataset = BioArcDataset(num_samples=500, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Setup Model
    model = TemporalMultimodalCBM(num_static_features=10, num_classes=2).to(DEVICE)
    
    # 3. Setup Loss & Optimizer
    criterion = MaskedJointLoss(lambda_c=0.5) # lambda controls concept importance
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 4. Training Loop
    model.train()
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for batch in loop:
            # Move data to device
            img = batch['img'].to(DEVICE)
            iop = batch['iop'].to(DEVICE)
            static = batch['static'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Move dicts to device
            c_true = {k: v.to(DEVICE) for k, v in batch['concepts'].items()}
            masks = {k: v.to(DEVICE) for k, v in batch['masks'].items()}
            
            # Forward Pass
            optimizer.zero_grad()
            predictions, c_pred = model(img, iop, static)
            
            # Calculate Loss
            loss, l_task, l_concept = criterion(predictions, c_pred, targets, c_true, masks)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), task=l_task.item(), concept=l_concept.item())
            
    print("Training Complete.")
    
    # 5. Save Model
    torch.save(model.state_dict(), "tm_cbm_model.pth")
    print("Model saved to tm_cbm_model.pth")

if __name__ == "__main__":
    train()
