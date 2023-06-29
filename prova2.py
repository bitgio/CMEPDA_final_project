"""
TITLE2
======
"""

import os
import pandoc
import numpy as np
import threading as thr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, InputLayer, Activation, Dropout
from scikit-learn.model_selection import train_test_split
from scikit-learn.metrics import roc_curve, auc
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import matlab.engine
eng = matlab.engine.start_matlab()

mammo_o = []
mammo_f = []
label = []
project_folder = "C:/Users/anapascual/exam_project/dataset/"
os.chdir(project_folder)
l = os.listdir()

def create_dataset(ls, o_img, f_img, lbl):
    """Function calling the Matlab file in order to filter the images
    
    ARGUMENTS
    ---------
    ls : list
         Chunk of files' directory.
         
    
    Return:
        Dataset with all images filtered.
    """
    if "_1_resized.pgm" in l:
        mo, mf = eng.dataset_filtered(eng.char(os.path.join(project_folder,l)), nargout = 2)
        o_img.append(mo)
        f_img.append(mf)
        lbl.append(1)
    elif "_2_resized.pgm" in l:
        mo, mf = eng.dataset_filtered(eng.char(os.path.join(project_folder,l)), nargout = 2)
        o_img.append(mo)
        f_img.append(mf)
        lbl.append(0)


os.chdir("C:/Users/anapascual/exam_project/")
threads = []
chunk = 49
print(chunk)
for i in range(6):
    t = thr.Thread(target = create_dataset, args = (l[i*chunk : (i+1)*chunk], mammo_o, mammo_f, label))
    threads.append(t)
    t.start()

for i in threads:
    i.join()

eng.quit()

mammo_o = np.asarray(mammo_o, dtype = 'float32')/255.
mammo_f = np.asarray(mammo_f, dtype = 'float32')/255.
label = np.asarray(label)

mammo_o_4d = np.reshape(mammo_o, (147, 125, 125, 1))
print(mammo_o_4d.shape)
mammo_f_4d = np.reshape(mammo_f, (147, 64, 64, 1))
print(mammo_f_4d.shape)

def cnn_o(shape=(125, 125, 1)):
    model = Sequential([
        
        Conv2D(5, (5,5), padding = 'same', input_shape = shape),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.2),
        
        
        Conv2D(6, (5,5), padding = 'same'),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.1),
        
        
        Conv2D(10, (5,5), padding = 'same'),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.1),
        
        Flatten(),
        
        Dense(10, activation = 'relu'),
        #Dropout(0.1),
        Dense(1, activation = 'sigmoid')        
        
    ])
    return model


model_o = cnn_o()
model_o.summary()

from tensorflow.keras.optimizers import SGD
learning_rate = 0.001
model_o.compile(optimizer = SGD(learning_rate, momentum = 0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])

reduce_on_plateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0)

acc = traino.history['accuracy']
val_acc = traino.history['val_accuracy']
loss = traino.history['loss']
val_loss = traino.history['val_loss']
    
epochs_range = range(1, len(acc)+1)
    #Train and validation accuracy 
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
    #Train and validation loss 
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

_, val_acc = model_o.evaluate(X_val_o, Y_val_o, verbose=0)
print('Validation accuracy: %.3f' % (val_acc))
    
    
preds = model_o.predict(X_val_o, verbose=1)

    #Compute Receiver operating characteristic (ROC)
fpr, tpr, _ = roc_curve(Y_val_o, preds)

roc_auc = auc(fpr, tpr)
   #Plot of a ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

