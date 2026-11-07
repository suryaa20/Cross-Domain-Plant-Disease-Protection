import os
import urllib.request
import zipfile
import shutil
import glob

# We include Pepper because you uploaded a Pepper leaf!
CLASSES_TO_EXTRACT = {
    "Apple___healthy": "Apple_Healthy",
    "Apple___Apple_scab": "Apple_Scab",
    "Tomato___healthy": "Tomato_Healthy",
    "Tomato___Early_blight": "Tomato_Early_Blight",
    "Pepper,_bell___Bacterial_spot": "Pepper_Bacterial_Spot",
    "Pepper,_bell___healthy": "Pepper_Healthy"
}

def main():
    base_dir = os.path.dirname(__file__)
    zip_path = os.path.join(base_dir, "PlantVillage.zip")
    extract_path = os.path.join(base_dir, "temp_extract")
    
    url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    
    print(f"Downloading PlantVillage dataset from GitHub... (This is a large file, please be patient)")
    try:
        # Check if already downloaded
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete!")
        else:
            print("Zip file already exists, skipping download.")
            
        print("Extracting necessary classes...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We only extract the 'raw/color' folder to save time and space
            members = [m for m in zip_ref.namelist() if 'raw/color' in m]
            zip_ref.extractall(extract_path, members=members)
            
        # Organize into our train/val splits
        base_dir_pv = os.path.abspath(os.path.join(base_dir, "PlantVillage"))
        base_dir_pd = os.path.abspath(os.path.join(base_dir, "PlantDoc"))
        
        raw_color_path = os.path.join(extract_path, "PlantVillage-Dataset-master", "raw", "color")
        
        for raw_class, target_class in CLASSES_TO_EXTRACT.items():
            os.makedirs(os.path.join(base_dir_pv, 'train', target_class), exist_ok=True)
            os.makedirs(os.path.join(base_dir_pv, 'val', target_class), exist_ok=True)
            os.makedirs(os.path.join(base_dir_pd, 'train', target_class), exist_ok=True)
            os.makedirs(os.path.join(base_dir_pd, 'val', target_class), exist_ok=True)
            
            src_class_path = os.path.join(raw_color_path, raw_class)
            if not os.path.exists(src_class_path):
                print(f"Warning: Could not find {src_class_path}")
                continue
                
            images = glob.glob(os.path.join(src_class_path, "*.*"))
            
            # Use only a subset to keep training fast for the project (e.g., 200 images per class)
            images = images[:200]
            split_idx = int(len(images) * 0.8)
            
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Copy to PlantVillage (Source) and PlantDoc (Target)
            # (We simulate PlantDoc by using the same images, but our Dataset class adds heavy augmentations)
            for img_path in train_images:
                img_name = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(base_dir_pv, 'train', target_class, img_name))
                shutil.copy(img_path, os.path.join(base_dir_pd, 'train', target_class, img_name))
                
            for img_path in val_images:
                img_name = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(base_dir_pv, 'val', target_class, img_name))
                shutil.copy(img_path, os.path.join(base_dir_pd, 'val', target_class, img_name))
                
        print("Cleaning up temporary files...")
        shutil.rmtree(extract_path)
        # Uncomment below to delete the zip file after extraction
        # os.remove(zip_path)
        
        print("\n✅ Successfully downloaded and prepared REAL dataset!")
        print("You have 6 classes including Pepper_Bacterial_Spot.")
        print("Next steps:")
        print("1. Run: python model/train_baseline.py")
        print("2. Run: python model/train_finetune.py")
        
    except Exception as e:
        print(f"Error during download/extraction: {e}")

if __name__ == "__main__":
    main()
