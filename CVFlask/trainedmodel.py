import numpy as np 
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

# Import libraries for model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import load_model

model = load_model("RetinalDiseaseCNN.h5")  
labels = ['glaucoma','cataract','normal','diabetic_retinopathy']

# load image
def loadimage(test):
    # Create a dataframe with the file path and the labels
    disease_type = [test]
    testdf = pd.DataFrame()
    for types in disease_type:
        for imagepath in tqdm(list(types.iterdir()), desc= str(types)):
            testdf = pd.concat([testdf, pd.DataFrame({'image': [str(imagepath)],'disease_type': "NIL"})], ignore_index=True)
            
    datagentest = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    test_data = datagentest.flow_from_dataframe(dataframe=testdf,
                                          x_col ='image',
                                          y_col = 'disease_type',
                                          target_size=(224,224))
    
    testdataset = np.concatenate([test_data.next()[0] for i in range(test_data.__len__())])
    return testdataset
         
# predict image
def predictimage(test):
    y_pred = model.predict(test)
    probdf = pd.DataFrame(y_pred, columns = labels)
    output = probdf.apply(lambda x: x*100)
    return output.to_html()
