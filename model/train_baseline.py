import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import get_dataloaders

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, device='cpu', save_path='baseline_model.pth'):
    """
    Standard PyTorch training loop.
    """
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', flush=True)
            
            # Deep copy the model if we get a better validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                
    print(f'Best Val Acc: {best_acc:4f}')
    return model

def main():
    # 1. Setup Data
    # We train the baseline EXCLUSIVELY on the lab-controlled PlantVillage dataset.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'PlantVillage')
    
    # We use a small batch size to ensure it runs on laptops
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size=16)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    num_classes = len(class_names)
    
    # 2. Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 3. Initialize Model
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # 4. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # We use Adam optimizer with a standard learning rate for transfer learning
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Baseline Training on PlantVillage...", flush=True)
    # Using 1 epoch for speed in demonstration. Increase to 10-20 for real training.
    train_model(model, dataloaders, criterion, optimizer, num_epochs=1, device=device)
    print("Training complete. Baseline model saved as 'baseline_model.pth'.", flush=True)

if __name__ == '__main__':
    main()
