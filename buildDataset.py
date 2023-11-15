# Import libraries
import numpy as np 
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def buildDataset():
    # Import each path of the image classes
    glaucoma = Path('./dataset/glaucoma')
    cataract = Path('./dataset/cataract')
    normal = Path('./dataset/normal')
    diabetic_retinopathy = Path('./dataset/diabetic_retinopathy')

    # Create a dataframe with the file path and the labels
    disease_type = [glaucoma, cataract, normal, diabetic_retinopathy]
    df = pd.DataFrame()
    for types in disease_type:
        for imagepath in tqdm(list(types.iterdir()), desc= str(types)):
            df = pd.concat([df, pd.DataFrame({'image': [str(imagepath)],'disease_type': [disease_type.index(types)]})], ignore_index=True)

    # Map the labels to the disease type
    df['disease_type'] = df['disease_type'].map({0:'glaucoma', 1:'cataract', 2:'normal', 3:'diabetic_retinopathy'})

    # Randomizing the dataset
    df1 = df.sample(frac=1).reset_index(drop=True)

    # Augumentation of images
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

    # Create train data
    train_data = datagen.flow_from_dataframe(dataframe=df1,
                                            x_col ='image',
                                            y_col = 'disease_type',
                                            target_size=(224,224),
                                            class_mode = 'categorical',
                                            batch_size = 32,
                                            shuffle = True,
                                            subset = 'training')

    # Create validation data
    valid_data = datagen.flow_from_dataframe(dataframe=df1,
                                            x_col ='image',
                                            y_col = 'disease_type',
                                            target_size=(224,224),
                                            class_mode = 'categorical',
                                            batch_size = 32,
                                            shuffle = False,
                                            subset = 'validation')

    # train_data.reset()
    x_train = np.concatenate([train_data.next()[0] for i in range(train_data.__len__())])
    y_train = np.concatenate([train_data.next()[1] for i in range(train_data.__len__())])
    print(x_train.shape)
    print(y_train.shape)

    # valid_data.reset()
    x_test = np.concatenate([valid_data.next()[0] for i in range(valid_data.__len__())])
    y_test = np.concatenate([valid_data.next()[1] for i in range(valid_data.__len__())])
    print(x_test.shape)
    print(y_test.shape)

    # labels = [key for key in train_data.class_indices]
    num_classes = len(disease_type)

    return x_train, y_train, train_data, x_test, y_test, valid_data, num_classes

if __name__ == '__main__':
    buildDataset()