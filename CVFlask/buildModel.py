# Import libraries for model
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D

def buildModel(num_classes):
    model = Sequential()

    model.add(Conv2D(64, (3,3), padding='same', input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
            
    model.add(Flatten())
            
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
            
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
            
    model.add(Dense(num_classes, activation='softmax'))

    # model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy']) 
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00025), metrics=['accuracy'])
    # model.summary()

    return model

if __name__ == '__main__':
    buildModel()