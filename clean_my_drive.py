"""
clean_my_drive.py
The operational inference engine. Integrates a deterministic OpenCV Laplacian 
gate with the MobileNetV2 deep learning architecture to generate a JSON 
manifest of degraded photos for web-based review.
"""

import os
import json
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def calculate_laplacian_variance(image_path, blur_threshold=100.0):
    """Executes the deterministic baseline filter via the 2nd spatial derivative."""
    image = cv2.imread(image_path)
    if image is None:
        return False # Categorize unreadable corrupted blocks as degraded
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var >= blur_threshold

def initialize_dl_pipeline(model_path):
    """Loads the saved Keras model from disk into operational memory."""
    return tf.keras.models.load_model(model_path)

def preprocess_image_for_tf(image_path):
    """Recreates the exact normalization constraints required by the CNN."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # OpenCV decodes BGR, model expects RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    
    # Statistical ImageNet alignment
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Expand tensor dimensions to represent a batch size of 1
    return np.expand_dims(image, axis=0)

def infer_deep_learning(image_path, model):
    """Executes the forward pass through the MobileNetV2 architecture."""
    input_tensor = preprocess_image_for_tf(image_path)
    if input_tensor is None:
        return True
    
    # Execute network prediction silently
    probabilities = model.predict(input_tensor, verbose=0)
    prediction = np.argmax(probabilities, axis=1)
    
    # Class Mapping: 0 represents Pristine, 1 represents Degraded
    is_bad = prediction[0] == 1 
    return is_bad

def generate_scan_report(target_dir, model_path, blur_threshold=100.0):
    """Initiates the recursive directory scan and builds a JSON manifest."""
    model = initialize_dl_pipeline(model_path)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    scan_results = []
    all_files = []
    
    # Recursively aggregate target paths
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                all_files.append(os.path.join(root, file))
                
    print(f"Initiating hybrid scan of {len(all_files)} files in '{target_dir}'...")
    
    for file_path in tqdm(all_files, desc="Scanning Gallery"):
        # Phase 1: High-Speed Mathematical Heuristic
        is_sharp = calculate_laplacian_variance(file_path, blur_threshold)
        
        if not is_sharp:
            scan_results.append({
                "id": file_path, # Using path as a unique ID for React mapping
                "path": file_path, 
                "reason": "Severe Blur (Failed Laplacian Gate)"
            })
        else:
            # Phase 2: High-Complexity Semantic Evaluation
            is_bad = infer_deep_learning(file_path, model)
            if is_bad:
                scan_results.append({
                    "id": file_path,
                    "path": file_path, 
                    "reason": "Accidental/Degraded Composition (Failed CNN)"
                })
                
    # Save the manifest to the project root
    output_file = 'scan_results.json'
    with open(output_file, 'w') as f:
        json.dump(scan_results, f, indent=4)
        
    print(f"\nExecution Terminated. Identified {len(scan_results)} degraded photos.")
    print(f"Report saved securely to: {os.path.abspath(output_file)}")

    return scan_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Deep Learning and OpenCV Photo Scanner")
    parser.add_argument("target_directory", type=str, help="Absolute path to the messy gallery you want to scan.")
    parser.add_argument("--model_path", type=str, default="models/weights/best_mobilenetv2.keras", help="Serialization path to the trained Keras model.")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Laplacian variance tuning parameter for the Phase 1 gate.")
    args = parser.parse_args()
    
    generate_scan_report(args.target_directory, args.model_path, args.blur_threshold)