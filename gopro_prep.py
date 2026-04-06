import os
import shutil
from tqdm import tqdm

# Define the source (where you just extracted GoPro)
# Adjust this if your extraction folder is named slightly differently
GOPRO_SOURCE = os.path.join("GoPro") 

# Define the target pipeline directories
TARGET_DIRS = {
    'train': {
        'good': os.path.join("data", "raw", "train", "good"),
        'bad': os.path.join("data", "raw", "train", "bad")
    },
    'val': {
        'good': os.path.join("data", "raw", "val", "good"),
        'bad': os.path.join("data", "raw", "val", "bad")
    }
}

# Ensure all target directories exist
for split in TARGET_DIRS.values():
    os.makedirs(split['good'], exist_ok=True)
    os.makedirs(split['bad'], exist_ok=True)

def sort_dataset():
    print("Scanning GoPro dataset...")
    files_to_move = []

    # Recursively find all images in the GoPro directory
    for root, _, files in os.walk(GOPRO_SOURCE):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                files_to_move.append(source_path)

    print(f"Found {len(files_to_move)} images. Reorganizing...")

    moved_count = 0
    for source_path in tqdm(files_to_move, desc="Routing Images"):
        # Determine if this image belongs in train or val (GoPro uses 'test')
        split_key = 'val' if 'test' in source_path.lower() else 'train'
        
        # Determine if this image is pristine (sharp) or degraded (blur)
        if 'sharp' in source_path.lower():
            class_key = 'good'
        elif 'blur' in source_path.lower() and 'blur_gamma' not in source_path.lower():
            class_key = 'bad'
        else:
            # Skip any weird extraneous files
            continue

        # Create a unique filename so frames from different video sequences don't overwrite each other
        # e.g., GOPR0384_11_00_000001.png
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(source_path)))
        new_filename = f"{parent_folder}_{os.path.basename(source_path)}"
        
        target_path = os.path.join(TARGET_DIRS[split_key][class_key], new_filename)

        # Move the file (use shutil.copy if you want to keep the original GoPro folder intact)
        shutil.move(source_path, target_path)
        moved_count += 1

    print(f"\nSuccessfully routed {moved_count} images into the pipeline structure.")

if __name__ == "__main__":
    if not os.path.exists(GOPRO_SOURCE):
        print(f"Error: Could not find the GoPro source directory at {GOPRO_SOURCE}")
        print("Please check your folder names.")
    else:
        sort_dataset()