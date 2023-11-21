# Import libraries
import os
from buildDataset import *
from buildModel import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import classification_report, confusion_matrix

def trainModel():
    x_train, y_train, train_data, x_test, y_test, valid_data, num_classes = buildDataset()

    model = buildModel(num_classes)

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

    # # Define learning rate = 0.00025
    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
    #                               factor=0.2,
    #                               patience=8,
    #                               verbose=1,
    #                               min_delta=0.00025)
    #                               # min_delta=0.0001)

    # Fitting the model
    history = model.fit(train_data,
                        validation_data=valid_data, 
                        epochs=15,
                        callbacks=[early_stopping, model_checkpoint])

    # training_accuracy = history.history['accuracy']
    # validation_accuracy = history.history['val_accuracy']

    # training_loss = history.history['loss']
    # validation_loss = history.history['val_loss']

    # plt.figure(figsize=(12, 4))

    # # Plotting training and validation accuracy
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, len(training_accuracy) + 1), training_accuracy, label='Training Accuracy')
    # plt.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, label='Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Epoch')
    # plt.legend()

    # # Plotting training and validation loss
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
    # plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss vs. Epoch')
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig('6585.svg', format='svg')
    # plt.show()

    # # Evaluate the model
    # y_pred = model.predict(x_test)

    # # Generate classification report of the model
    # labels = ['glaucoma','cataract','normal','diabetic_retinopathy']

    # y_pred_argmax = [np.argmax(prob) for prob in y_pred]
    # y_pred_binary = [[1 if i == n else 0 for i in range(len(labels))] for n in y_pred_argmax]

    # # report = classification_report(y_test,y_pred,target_names = labels)

    # report = classification_report(y_test, y_pred_binary, target_names=labels, zero_division=0)

    # print(report)

    # conf_matrix = confusion_matrix([np.argmax(y) for y in y_test], [np.argmax(y) for y in y_pred])

    # plt.figure(figsize=(8, 6))

    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=labels, yticklabels=labels)

    # plt.xlabel('Prediction')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')

    # plt.savefig('6585-cm.svg', format='svg')
    # plt.show()

    return history

if __name__ == '__main__':
    trainModel()