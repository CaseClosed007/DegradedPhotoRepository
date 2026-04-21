"""
src/train.py
Research Implementation: MobileNetV2 backbone augmented with a Custom Spatial 
Attention Mechanism and optimized via Binary Focal Crossentropy to prioritize 
hard-to-detect sensor degradations.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Import the DataLoader pipeline from your data_prep script
from data_prep import create_dataloaders

def build_attention_model():
    """
    Instantiates the MobileNetV2 architecture and injects a novel 
    Spatial Attention Block prior to global pooling.
    """
    # Fetch optimal pre-trained weights for feature extraction
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the initial feature extraction blocks to preserve generalized edge mapping
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model.output
    
    # --- NOVEL CONTRIBUTION: Spatial Attention Block ---
    # We compress the channel depth to 1, creating a 2D map of "importance"
    # This teaches the network WHERE to look for isolated sensor noise or blur
    attention_map = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same', name='attention_map')(x)
    
    # Multiply the original features by this mathematical attention map
    x = Multiply(name='attention_multiplication')([x, attention_map])
    # ---------------------------------------------------

    # Reconfigure the final pooling and classification layers
    x = GlobalAveragePooling2D()(x)
    
    # Using a single neuron with Sigmoid for optimal Binary Focal Loss compatibility
    predictions = Dense(1, activation='sigmoid', name='final_classifier')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(train_dir, val_dir, epochs=15, batch_size=32, lr=3e-4):
    """
    Executes the training loop using Advanced Focal Loss to penalize
    easy classifications and force gradient updates on hard edge cases.
    """
    # Enable mixed precision to accelerate training on compatible hardware
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"Executing on GPU: {bool(tf.config.list_physical_devices('GPU'))}")

    # Build the novel computational graph
    model = build_attention_model()

    # Instantiate the heavily optimized tf.data.Dataset pipelines
    print("Building Data Pipelines...")
    train_dataset = create_dataloaders(train_dir, batch_size=batch_size, is_training=True)
    val_dataset = create_dataloaders(val_dir, batch_size=batch_size, is_training=False)

    # AdamW incorporates decoupled weight decay to enforce feature generalization
    optimizer = AdamW(learning_rate=lr, weight_decay=1e-2)

    # --- NOVEL CONTRIBUTION: Focal Loss ---
    # Gamma = 2.0 aggressively reduces the loss for easy examples
    # Alpha = 0.25 balances the ratio between the Pristine and Degraded classes
    focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, 
        alpha=0.25, 
        gamma=2.0
    )

    model.compile(
        optimizer=optimizer,
        loss=focal_loss,
        metrics=['accuracy']
    )

    # Create the directory for saving the serialized model
    os.makedirs('models/weights', exist_ok=True)

    # Ensure only the mathematically optimal topology configuration is written to disk
    checkpoint_cb = ModelCheckpoint(
        filepath='models/weights/best_attention_mobilenet.keras',
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
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, lr_scheduler]
    )

if __name__ == "__main__":
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