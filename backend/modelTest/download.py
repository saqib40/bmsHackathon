import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.callbacks import EarlyStopping

# Set up Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset
dataset_name = "jxwleong/coral-reef-dataset"
dataset_path = api.dataset_download_files(dataset_name, path='./dataset', unzip=True)
print("Dataset downloaded to:", dataset_path)

# Define preprocess_data function
def preprocess_data(dataset_dir):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Define create_coral_health_model function
def create_coral_health_model(num_classes):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Preprocess data
train_generator, validation_generator = preprocess_data('./dataset/coral-reef-dataset')

# Build and train model
EPOCHS = 20
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = create_coral_health_model(num_classes=3)
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Save model
model.save('coral_health_model.h5')
