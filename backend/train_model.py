import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from sklearn.model_selection import train_test_split

# Configuration
DATASET_PATH = 'dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Data Preprocessing
def preprocess_data(dataset_path):
    # Load and preprocess images
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 20% for validation
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Build Model
def create_coral_health_model(num_classes):
    base_model = ResNet50V2(
        weights='imagenet', 
        include_top=False, 
        input_shape=IMAGE_SIZE + (3,)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Training Function
def train_coral_model():
    train_generator, validation_generator = preprocess_data(DATASET_PATH)
    
    num_classes = len(train_generator.class_indices)
    model = create_coral_health_model(num_classes)

    # Callback for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # Model Training
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    # Save Model
    model.save('coral_health_model.h5')
    print("Model saved successfully!")

    return model, history

# Execute Training
if __name__ == '__main__':
    model, training_history = train_coral_model()

    # Optional: Plot training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['accuracy'], label='Training Accuracy')
    plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_history.history['loss'], label='Training Loss')
    plt.plot(training_history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()