# Import libraries for model
import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D, Input, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
import tensorflow as tf


def conv4LogModel():
    input_layer = Input(shape=(224,224,3))

    conv1 = Conv2D(64, (3,3), padding='same')(input_layer)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    pool1 = MaxPooling2D((2,2))(act1)
    drop1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(128, (5,5), padding='same')(drop1)
    batch2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(batch2)
    pool2 = MaxPooling2D((2,2))(act2)
    drop2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(512, (3,3), padding='same')(drop2)
    batch3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(batch3)
    pool3 = MaxPooling2D((2,2))(act3)
    drop3 = Dropout(0.25)(pool3)

    conv4 = Conv2D(512, (3,3), padding='same')(drop3)
    batch4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(batch4)
    pool4 = MaxPooling2D((2,2))(act4)
    drop4 = Dropout(0.25)(pool4)
            
    flat = Flatten()(drop4)
            
    dense1 = Dense(256)(flat)
    dense_batch1 = BatchNormalization()(dense1)
    dense_act1 = Activation('relu')(dense_batch1)
    dense_drop1 = Dropout(0.25)(dense_act1)
            
    dense2 = Dense(512)(dense_drop1)
    dense_batch2 = BatchNormalization()(dense2)
    dense_act2 = Activation('relu')(dense_batch2)
    dense_drop2 = Dropout(0.25)(dense_act2)

    # model = Model(input_layer, dense_drop2)

    log1 = Dense(1, activation='sigmoid')(dense_drop2)
    log2 = Dense(1, activation='sigmoid')(dense_drop2)
    log3 = Dense(1, activation='sigmoid')(dense_drop2)
    log4 = Dense(1, activation='sigmoid')(dense_drop2)
    output_layer = concatenate([log1, log2, log3, log4], axis=-1)
    # output_layer = concatenate([log1, log2, log3], axis=-1)

    model = Model(input_layer, output_layer)

    model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy']) 
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.00025), metrics=['accuracy'])
    model.summary()

    return model

def build4LogDataset():
    # def makeLabel(idx, num_classes):
    #     label = np.zeros(num_classes - 1)
    #     if idx == (num_classes - 1):
    #         return label
    #     else:
    #         label[idx] = 1
    #         return label

    glaucoma = Path('./dataset/glaucoma')
    cataract = Path('./dataset/cataract')
    normal = Path('./dataset/normal')
    diabetic_retinopathy = Path('./dataset/diabetic_retinopathy')

    # Create a dataframe with the file path and the labels
    disease_type = [glaucoma, cataract, diabetic_retinopathy, normal]
    df = pd.DataFrame()
    for types in disease_type:
        for imagepath in tqdm(list(types.iterdir()), desc= str(types)):
          # label = makeLabel(disease_type.index(types), len(disease_type))
          # label = tf.convert_to_tensor(label)
          # df = pd.concat([df, pd.DataFrame({'image': [str(imagepath)],'label': [label]})], ignore_index=True)
          df = pd.concat([df, pd.DataFrame({'image': [str(imagepath)],'label': [disease_type.index(types)]})], ignore_index=True)

    # df['label'] = df['label'].map({0:'normal', 1:'glaucoma', 2:'cataract', 3:'diabetic_retinopathy'})

    # Randomizing the dataset
    df1 = df.sample(frac=1).reset_index(drop=True)

    # Produce test set
    X = df1['image']
    y = df1['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Augumentation of images
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

    # Create train data
    train_data = datagen.flow_from_dataframe(dataframe=train_df,
                                            x_col ='image',
                                            y_col = 'label',
                                            target_size=(224,224),
                                            class_mode = 'raw',
                                            batch_size = 32,
                                            shuffle = True,
                                            subset = 'training')

    # Create validation data
    valid_data = datagen.flow_from_dataframe(dataframe=train_df,
                                            x_col ='image',
                                            y_col = 'label',
                                            target_size=(224,224),
                                            class_mode = 'raw',
                                            batch_size = 32,
                                            shuffle = False,
                                            subset = 'validation')
    
    # Create test data
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0)

    test_data = datagen.flow_from_dataframe(dataframe=test_df,
                                            x_col ='image',
                                            y_col = 'label',
                                            target_size=(224,224),
                                            class_mode = 'raw',
                                            batch_size = 32,
                                            shuffle = False,
                                            subset = 'training')

    return train_data, valid_data, test_data

def train4Logistic(model, dataset):
    train_data, valid_data, test_data= dataset

    # Define checkpoint
    # filepath = "RetinalDiseaseCNN.h5"
    checkpoint_path = "checkpoint/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       save_best_only=True,
                                       save_weights_only=True, 
                                       monitor='val_accuracy', 
                                       mode='max', 
                                       verbose=1)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', 
                                patience=8, 
                                verbose=1)
                                # restore_best_weights=True)

    # Fitting the model
    history = model.fit(train_data,
                        validation_data=valid_data, 
                        epochs=15,
                        callbacks=[early_stopping, model_checkpoint])
    
    model.save('RetinalDisease4Log.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print(acc, val_acc, loss, val_loss)

    return history, model

# run the model:
# model = conv4LogModel()
# dataset = build4LogDataset()
# history, model = train4Logistic(model, dataset)
# train_data, valid_data, test_data= dataset
# scores = model.evaluate(test_data)

# look at outputs:
# predictions = model.predict(test_data)
# print(predictions)
# print(np.sum(predictions, axis=1))