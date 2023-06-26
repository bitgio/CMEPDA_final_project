import os
import numpy as np
import threading as thr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, InputLayer, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matlab.engine



def create_dataset(ls):
    """ Function calling the Matlab file in order to filter the images
    
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


if __name__ == '__main__':

    mammo_o = []
    mammo_f = []
    label = []
    project_folder = "C:/Users/anapascual/cmepda/progetto/dataset/"
    os.chdir(project_folder)
    l = os.listdir()

    eng = matlab.engine.start_matlab()
    
    os.chdir("C:/Users/anapascual/cmepda/progetto/")
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

