import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model, freeze_backbone, unfreeze_all
from dataset import get_dataloaders
from train_baseline import train_model

def main():
    # 1. Setup Data
    # For Domain Adaptation, we fine-tune on the TARGET domain (PlantDoc).
    # In a real scenario, we'd only have a small labeled subset of this.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'PlantDoc')
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size=16)
    dataloaders = {'train': train_loader, 'val': val_loader}
    num_classes = len(class_names)
    
    # 2. Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 3. Load the Baseline Model
    baseline_path = '../baseline_model.pth'
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} not found. Please run train_baseline.py first.")
        return
        
    model = get_model(num_classes=num_classes, pretrained=False) # Architecture only
    model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
    model = model.to(device)
    
    # 4. Domain Adaptation Strategy: Two-Stage Fine-Tuning
    criterion = nn.CrossEntropyLoss()
    
    # --- STAGE 1: Train only the final layer ---
    # We freeze the backbone so we don't ruin the feature extractors 
    # learned from the large PlantVillage dataset.
    print("\n--- STAGE 1: Fine-tuning final layer on PlantDoc ---", flush=True)
    model = freeze_backbone(model)
    
    # Only optimize the fully connected layer
    optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=0.001)
    model = train_model(model, dataloaders, criterion, optimizer_stage1, num_epochs=10, device=device, save_path='../domain_adapted_model.pth')
    
    # --- STAGE 2: Unfreeze and train everything with a tiny learning rate ---
    # Now we let the whole network gently adapt to the new domain.
    print("\n--- STAGE 2: Full network Fine-Tuning ---", flush=True)
    model = unfreeze_all(model)
    
    # Use a much smaller learning rate (e.g., 1e-5) so we don't forget Source knowledge
    optimizer_stage2 = optim.Adam(model.parameters(), lr=1e-5)
    model = train_model(model, dataloaders, criterion, optimizer_stage2, num_epochs=5, device=device, save_path='../domain_adapted_model.pth')
    
    # 5. Save the Adapted Model
    torch.save(model.state_dict(), '../domain_adapted_model.pth')
    print("\nDomain Adaptation complete. Model saved as '../domain_adapted_model.pth'.", flush=True)

if __name__ == '__main__':
    main()
