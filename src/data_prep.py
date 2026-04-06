"""
src/data_prep.py
Provides the foundational dataset abstractions and the highly optimized 
Albumentations degradation pipeline to simulate optical and sensor failures.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

def get_training_transforms():
    """
    Constructs the synthetic light-path degradation matrix.
    Utilizes AdvancedBlur for Generalized Gaussian optical aberration and 
    ISONoise for read/shot noise emulation.
    """
    return A.Compose([
        # 1. Optical Aberration Simulation (Blur)
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=0.4), # Applied to 40% of the distribution

        # 2. Photonic Sensor and ADC Conversion Noise Simulation
        A.OneOf([
            A.ISONoise(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ], p=0.4), 

        # 3. Spatial Distortion simulating accidental captures
        A.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), p=0.3), 

        # 4. CRITICAL FIX: Guarantee Uniform Tensor Shape
        # If the crop above didn't trigger, this forces the image to 224x224. 
        # If the crop did trigger, this harmlessly leaves it at 224x224.
        A.Resize(height=224, width=224),

        # Normalization conforms the pixel statistics to ImageNet baselines
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ])

def get_validation_transforms():
    """
    Validation requires strict determinism to evaluate ground truth accurately; 
    therefore, only topological resizing and statistical normalization are applied.
    """
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def process_data(image_path, label, transform):
    """
    Decodes the image path, executes the C++ Albumentations pipeline,
    and structures the numpy array for TensorFlow ingestion.
    """
    def _process(path, lbl):
        path = path.decode('utf-8')
        image = cv2.imread(path)
        if image is None:
            # Silent fallback mechanism to prevent dataset crashes on corrupted files
            image = np.zeros((224, 224, 3), dtype=np.float32)
        else:
            # OpenCV decodes images in BGR format; strict conversion to RGB is required
            # for compatibility with standard pre-trained CNN weight expectations.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if transform:
            augmented = transform(image=image)
            image = augmented['image']
            
        return image.astype(np.float32), np.int32(lbl)
    
    # Wrap the python execution inside the TF graph
    image, label = tf.numpy_function(_process, [image_path, label], [tf.float32, tf.int32])
    
    # Explicitly set shapes to avoid dynamic shape errors in Keras models
    image.set_shape([224, 224, 3])
    label.set_shape([])
    
    return image, label

def create_dataloaders(split_dir, batch_size=32, is_training=True):
    """
    Traverses a specific split directory (train/val), collates the file paths, 
    maps them to binary integers, and constructs the tf.data pipeline.
    """
    image_paths = []
    labels = []

    # Dynamically locate the binary class folders inside the provided split directory
    good_dir = os.path.join(split_dir, 'good')
    bad_dir = os.path.join(split_dir, 'bad')

    # Map the topological structure to binary labels (0: Pristine, 1: Degraded)
    for directory, label in [(good_dir, 0), (bad_dir, 1)]:
        if not os.path.exists(directory):
            print(f"WARNING: '{directory}' not found in {split_dir}. You need both classes to train.")
            continue
            
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
                    labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    transform = get_training_transforms() if is_training else get_validation_transforms()
    
    # Apply multiprocessing mapped transformations
    dataset = dataset.map(lambda x, y: process_data(x, y, transform), num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    # Enable prefetching to prevent GPU starvation while CPU processes augmentations
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset