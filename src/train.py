"""
Training script for speech emotion recognition model.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model


def load_data(features_path='data/processed/features.npy', 
              labels_path='data/processed/labels.npy'):
    """Load preprocessed features and labels."""
    print("Loading data...")
    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"Loaded {len(features)} samples")
    print(f"Feature shape: {features.shape}")
    return features, labels


def prepare_data(features, labels, test_size=0.2):
    """Split data and convert labels to one-hot encoding."""
    print("\nPreparing data...")
    
    label_to_int = {label: idx for idx, label in enumerate(np.unique(labels))}
    int_labels = np.array([label_to_int[label] for label in labels])
    
    num_classes = len(label_to_int)
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_to_int}")
    
    y_categorical = to_categorical(int_labels, num_classes)
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, y_categorical, test_size=test_size, random_state=42, stratify=int_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val, num_classes


def train_model(X_train, X_val, y_train, y_val, num_classes, 
                epochs=50, batch_size=32):
    """Train the emotion recognition model."""
    print("\nBuilding model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_model(input_shape, num_classes)
    
    print("\nModel architecture:")
    model.summary()
    
    Path('models').mkdir(exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def save_history(history, path='models/training_history.pkl'):
    """Save training history to pickle file."""
    print(f"\nSaving training history to {path}")
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved!")


if __name__ == '__main__':
    features, labels = load_data()
    
    X_train, X_val, y_train, y_val, num_classes = prepare_data(features, labels)
    
    model, history = train_model(X_train, X_val, y_train, y_val, num_classes)
    
    save_history(history)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model saved to: models/best_model.h5")
    print(f"Training history saved to: models/training_history.pkl")
    print("="*50)
