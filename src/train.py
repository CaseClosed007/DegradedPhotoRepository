"""
src/train.py
Handles the configuration of the MobileNetV2 architecture, Keras compilation,
and the execution of the mixed-precision optimization loop.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Import the updated DataLoader pipeline from your data_prep script
from data_prep import create_dataloaders

def build_model(num_classes=2):
    """
    Instantiates the MobileNetV2 architecture via Keras Applications.
    Modifies the terminal classifier for binary classification via transfer learning.
    """
    # Fetch optimal pre-trained weights for feature extraction
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the initial feature extraction blocks to preserve generalized edge mapping
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Reconfigure the final pooling and classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(train_dir, val_dir, epochs=15, batch_size=32, lr=3e-4):
    """
    Executes the training loop, applying AdamW regularization, Learning Rate Annealing,
    Automatic Mixed Precision (AMP), and validation dataset monitoring.
    """
    # Enable mixed precision to accelerate training by utilizing 16-bit operations where safe
    # Note: On macOS, this utilizes the Metal Performance Shaders (MPS)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Verify the Mac's GPU is recognized by TensorFlow
    print(f"Executing on GPU: {bool(tf.config.list_physical_devices('GPU'))}")

    # Build the computational graph
    model = build_model(num_classes=2)

    # Instantiate the heavily optimized tf.data.Dataset pipelines for BOTH splits
    print("Building Data Pipelines...")
    train_dataset = create_dataloaders(train_dir, batch_size=batch_size, is_training=True)
    val_dataset = create_dataloaders(val_dir, batch_size=batch_size, is_training=False)

    # AdamW incorporates decoupled weight decay to enforce feature generalization
    optimizer = AdamW(learning_rate=lr, weight_decay=1e-2)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create the directory for saving the serialized model
    os.makedirs('models/weights', exist_ok=True)

    # Ensure only the mathematically optimal topology configuration is written to disk
    # We monitor validation loss (val_loss) to save the model that generalizes the best
    checkpoint_cb = ModelCheckpoint(
        filepath='models/weights/best_mobilenetv2.keras',
        save_best_only=True,
        monitor='val_loss', 
        mode='min',
        verbose=1
    )

    # Gently decays the learning rate to settle into narrow local minima if training stalls
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=2,
        verbose=1
    )

    print("\nInitiating optimization routine...")
    model.fit(
        train_dataset,
        validation_data=val_dataset, # Evaluates unseen data at the end of every epoch
        epochs=epochs,
        callbacks=[checkpoint_cb, lr_scheduler]
    )

if __name__ == "__main__":
    # Assumes execution from the project root (e.g., python src/train.py)
    # Update these paths if your root folder execution path varies
    TRAIN_DIRECTORY = 'data/raw/train'
    VALIDATION_DIRECTORY = 'data/raw/val'
    
    if not os.path.exists(TRAIN_DIRECTORY) or not os.path.exists(VALIDATION_DIRECTORY):
        print("Error: Could not locate the training or validation directories.")
        print("Ensure you are running this script from the project root and that your data is sorted.")
    else:
        train_model(
            train_dir=TRAIN_DIRECTORY, 
            val_dir=VALIDATION_DIRECTORY
        )