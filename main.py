import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt

# import numpy as np
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

from utilities import *

# Constants
BATCH_SIZE=64
IMAGE_SIZE=224
CHANNELS=3

# Insert custom path here
custom_path = ''

# Reading csv
train_csv_path = custom_path + '/Training_Set/Training_Set/RFMiD_Training_Labels.csv'
train_ds_csv=pd.read_csv(train_csv_path)

# Creating dataset path
base_path = custom_path + '/dataset/'

if __name__ == "__main__":
    directory_names = list(train_ds_csv.columns[2:])
    directory_names.append('NORMAL')

    # Create directories
    for directory_name in directory_names:
        directory_path = os.path.join(base_path, directory_name)
        try:
            os.makedirs(directory_path)
            # print(f"Directory '{directory_name}' created at '{directory_path}'")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists at '{directory_path}'")

    move('Training')
    print("Done Training")
    move('Test')
    print("Done Testing")
    move('Evaluation')
    print("Done Evaluation")

    # Delete subfolders with fewer than 30 images in each directory
    for subdir, _, _ in os.walk(base_path):
        delete_folders(subdir)

    # Specify the root folder containing subdirectories with images
    root_folder_path = base_path

    delete_extra_images(root_folder_path)

    # Split dataset to train_ds, val_ds, test_ds
    dataset=tf.keras.preprocessing.image_dataset_from_directory(
        "dataset", shuffle=True,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.experimental.preprocessing.RandomContrast(factor=0.1),  # Adjust contrast
        # layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Random translation
        layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),  # Random zoom
        # layers.experimental.preprocessing.Normalization(mean=0.5, variance=0.5)  # Normalization
    ])
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2)
    ])

    # CNN model structure
    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = len(os.listdir(base_path))
    model = models.Sequential([
        resize_and_rescale,
        # data_augmentation,
        layers.Conv2D(256, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        # layers.Dropout(0.25),  # Dropout layer added after the first Conv2D layer
        layers.Conv2D(256, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(512, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        # layers.Dropout(0.25),  # Dropout layer added after the third Conv2D layer
        layers.Conv2D(512, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(512, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        # layers.Dropout(0.25),  # Dropout layer added after the fifth Conv2D layer
        # layers.Conv2D(1024, kernel_size=(3,3), activation='relu'),
        # layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # layers.Dropout(0.25),  # Dropout layer added before the final Dense layer
        layers.Dense(n_classes, activation='softmax')
    ])

    # Building model
    model.build(input_shape=input_shape)
    # model.summary()

    # Training model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    EPOCHS=300
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
        validation_data=val_ds
    )

    # Evaluating model accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    axis_size=len(acc)
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.plot(range(axis_size), acc, label='Training Accuracy')
    plt.plot(range(axis_size), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Evaluate the model
    scores = model.evaluate(test_ds)
    print(f'Test accuracy: {scores}')

    # Empty lists for predictions and true labels
    # predicted_labels = []
    # true_labels = []

    # # Make predictions on the test set
    # for images, labels in test_ds:
    #     predictions = model(images)
    #     predicted_labels.extend(np.argmax(predictions, axis=1))  # Obtaining the class with maximum probability as the predicted label
    #     true_labels.extend(labels)

    # # Convert lists to numpy arrays
    # predicted_labels = np.array(predicted_labels)
    # true_labels = np.array(true_labels)

    # # Calculate confusion matrix
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # # print("Confusion Matrix:")
    # # print(conf_matrix)

    # # Plot the confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.title("Confusion Matrix")
    # plt.show()

    # model.save('RetinalDiseaseCNN')