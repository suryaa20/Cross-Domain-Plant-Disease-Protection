import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# DATA AUGMENTATION STRATEGY
# ==========================================
# The core goal of this project is cross-domain robustness.
# We use augmentations to artificially introduce "real-world" noise 
# into the clean PlantVillage data to bridge the domain gap.

# Transformations for TRAINING data
# We add blur, color jitter, and random crops to simulate 
# the messy, poorly lit conditions of PlantDoc (field images).
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Simulate different zoom levels
    transforms.RandomHorizontalFlip(),                   # Basic augmentation
    transforms.RandomRotation(30),                       # Leaves can be at any angle
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Lighting variation
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Simulate out-of-focus field shots
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

# Transformations for VALIDATION/TESTING data
# No random augmentations here. We just resize and normalize 
# so we get consistent evaluations.
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(data_dir, batch_size=32):
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    Args:
        data_dir: Path to the dataset (e.g., 'data/PlantVillage' or 'data/PlantDoc')
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory not found: {train_dir}")
        
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    
    test_loader = None
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Store class names for reference
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names
