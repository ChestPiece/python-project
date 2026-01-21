import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_utils import build_model

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = '../data'  # Assuming running from tracefake/models/
MODEL_SAVE_DIR = 'saved_model'
MODEL_NAME = 'tracefake_v1.h5'

def train():
    # Ensure data directories exist
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: Data directory not found at {train_dir}")
        print("Please structure your data as: data/train/fake, data/train/real")
        return

    # Data Augmentation and Generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # If no separate val folder
    )

    # Load Data
    print("Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR, # Point to parent 'data' if using classes subfolders directly inside
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    print(f"Classes: {train_generator.class_indices}")

    # Build Model
    print("Building Model...")
    model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model.summary()

    # Callbacks
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_SAVE_DIR, MODEL_NAME),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    print("Starting Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Training Complete.")

if __name__ == '__main__':
    train()
