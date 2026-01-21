import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    """
    Builds the CNN model using EfficientNetB0 as the base.
    Uses Transfer Learning.
    
    Args:
        input_shape: Tuple (height, width, channels).
        
    Returns:
        model: compiled Keras model.
    """
    # Load EfficientNetB0 without top layers
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    # Output layer: 1 neuron, sigmoid for binary classification (Real vs Fake)
    # 0 = Real, 1 = Fake (or vice versa depending on folder structure, usually alphabetical)
    # Let's assume: 0=Fake, 1=Real (Need to verify this with data generator class indices)
    # For now, we output probability of being class 1.
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_trained_model(model_path):
    """
    Loads a trained model from disk.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None
