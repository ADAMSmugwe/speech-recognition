"""
Deep learning model definitions for emotion recognition.
"""

from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense


def create_model(input_shape, num_classes):
    """
    Create a 1D-CNN model for speech emotion recognition.
    
    Args:
        input_shape: Shape of input features (n_mfcc, time_steps)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == '__main__':
    input_shape = (40, 130)
    num_classes = 8
    
    model = create_model(input_shape, num_classes)
    model.summary()
