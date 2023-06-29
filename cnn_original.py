import os
import numpy as np
import threading as thr
import matplotlib.pyplot as plt
from sklearn .model_selection import train_test_split
from sklearn .metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, InputLayer, Activation, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
import matlab.engine

def create_dataset(lista, o_img, f_img, labels):
    """Function calling the Matlab file in order to filter the images.
    
    Arguments
    ---------
    
    lista : list
        Chunk of file directories.
    
    Return:
        Dataset with all the images filtered.
    """
    
    for element in lista:
        if "_1_resized.pgm" in element:
            mo, mf = eng.dataset_filtered(eng.char(os.path.join(data_folder, element)), nargout = 2)
            o_img.append(mo)
            f_img.append(mf)
            labels.append(1)
        elif "_2_resized.pgm" in element:
            mo, mf = eng.dataset_filtered(eng.char(os.path.join(data_folder, element)), nargout = 2)
            o_img.append(mo)
            f_img.append(mf)
            labels.append(0)



def cnn_o(shape=(125, 125, 1)):
    """CNN original mammo model.
    
    Arguments
    ---------
    
    shape : tuple
        Size.
    
    Return:
        Dataset with all the images filtered.
    """
    
    model = Sequential([
        
        Conv2D(7, (4,4), padding = 'same', input_shape = shape),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.2),
        
        
        Conv2D(8, (4,4), padding = 'same'),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.1),
        
        
        Conv2D(10, (4,4), padding = 'same'),
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



if __name__ == '__main__':
    eng = matlab.engine.start_matlab()

    mammo_o, mammo_f, label = [], [], []
    data_folder = "C:/Users/anapascual/exam_project/dataset/"
    os.chdir(data_folder)
    l = os.listdir()

    os.chdir("C:/Users/anapascual/exam_project/")
    threads = []
    chunk = 6

    for i in range(49):
        t = thr.Thread(target = create_dataset, args = (l[i*chunk : (i+1)*chunk], mammo_o, mammo_f, label))
        threads.append(t)
        t.start()
        
    for j in threads:
        j.join()

    eng.quit()

    mammo_o = np.asarray(mammo_o, dtype = 'float32')/255.
    mammo_f = np.asarray(mammo_f, dtype = 'float32')/255.
    label = np.asarray(label)

    mammo_o_4d = np.reshape(mammo_o, (147, 125, 125, 1))
    mammo_f_4d = np.reshape(mammo_f, (147, 64, 64, 1))


