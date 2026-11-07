import os
import urllib.request
import zipfile
import numpy as np
from PIL import Image

# We define the common classes between PlantVillage (Source) and PlantDoc (Target)
# For simplicity in this academic project, we use 4 distinct classes.
COMMON_CLASSES = [
    "Apple_Healthy",
    "Apple_Scab",
    "Tomato_Healthy",
    "Tomato_Early_Blight"
]

def create_dummy_dataset(base_dir, num_images_per_class=20):
    """
    Creates dummy images for testing the pipeline without downloading GBs of data.
    In a real scenario, this script would download PlantVillage and PlantDoc.
    """
    print(f"Setting up dummy dataset at {base_dir}...")
    for split in ['train', 'val', 'test']:
        for cls in COMMON_CLASSES:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            for i in range(num_images_per_class):
                # Create a simple dummy image (random noise)
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Add some color bias to simulate leaves (more green)
                if "Healthy" in cls:
                    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 50, 0, 255) # Boost green
                elif "Scab" in cls or "Blight" in cls:
                    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 50, 0, 255) # Boost red/brown
                
                img = Image.fromarray(img_array)
                img.save(os.path.join(dir_path, f"img_{i}.jpg"))

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Setup Source Domain (PlantVillage - Lab style)
    pv_dir = os.path.join(base_path, "PlantVillage")
    create_dummy_dataset(pv_dir, num_images_per_class=50)
    
    # 2. Setup Target Domain (PlantDoc - Real world style)
    pd_dir = os.path.join(base_path, "PlantDoc")
    create_dummy_dataset(pd_dir, num_images_per_class=20)
    
    print("Data preparation complete!")
    print("NOTE: For the actual project, replace the dummy folders with actual PlantVillage and PlantDoc datasets, categorized into train/val/test folders.")

if __name__ == "__main__":
    main()
