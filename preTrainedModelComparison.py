#Imports
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import tensorflow as tf
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import EfficientNetB0
import os
import random
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator

def loadFromFolder(augPath, notAugPath, valPath, classID, n_samples, train_images, train_labels, test_images, test_labels):
    train_aug = random.choices(population=os.listdir(augPath), k=n_samples)
    train_notAug = random.choices(population=os.listdir(notAugPath), k=n_samples)
    ### load AUGMENTED samples in training set
    c = 0
    for img_path in train_aug:
        # LOG
        if verbose == 1:
            print(f"[LOG] loading {augPath+img_path}")
        img =  image.load_img(augPath+img_path, target_size=(HEIGHT, WIDTH),color_mode="grayscale")
        img_arr = image.img_to_array(img)
        train_images.append(img_arr)
        train_labels.append(classID)
        c+=1
    print(f"[INFO] loaded {c} images from {augPath}")
                  
    ### load NOT_AUGMENTED samples in training set
    c = 0
    for img_path in train_notAug:
        # LOG
        if verbose == 1:
            print(f"[LOG] loading {notAugPath+img_path}")
        img =  image.load_img(notAugPath+img_path, target_size=(HEIGHT, WIDTH),color_mode="grayscale")
        img_arr = image.img_to_array(img)
        train_images.append(img_arr)
        train_labels.append(classID)
        c+=1
    print(f"[INFO] loaded {c} images from {notAugPath}")
                  
    ### load VAL_DATA
    c = 0
    for img_path in os.listdir(valPath):
        #log 
        if verbose == 1:   
          print(f"[INFO] {valPath+img_path}")
        img =  image.load_img(valPath+img_path, target_size=(HEIGHT, WIDTH),color_mode="grayscale")
        img_arr = image.img_to_array(img)
        test_images.append(img_arr)
        test_labels.append(classID)
        c += 1
    print(f"[INFO] loaded {c} images from {valPath}") 
    


def load_images():
    for i in range(len(DATASET_LABELS)):  
        for j in range(len(FOLDER_LABELS)):
          #path = "/content/drive/MyDrive/AI_Project/datasets/"+DATASET_LABELS[i]+"/"+FOLDER_LABELS[j]+"/"
          path ="/kaggle/input/d/mattiacrispino/covid-datasets/datasets/"+DATASET_LABELS[i]+"/"+FOLDER_LABELS[j]+"/" 
          for img_path in os.listdir(path):
            img =  image.load_img(path+img_path, target_size=(HEIGHT, WIDTH),color_mode="grayscale")
            img_arr = image.img_to_array(img)
            # log
            if verbose == 1:
              print(f"[INFO] Opening {path+img_path}")

            # train data
            if i != 2:
              train_images.append(img_arr)
              train_labels.append(j+1)
            # test data
            else:  
              test_images.append(img_arr)
              test_labels.append(j+1)

    # LOAD OVERSAMPLED COVID-19
    path = "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/OversampledAugmentedCOVID-19/COVID-19/"
    #path = "/content/drive/MyDrive/AI_Project/datasets/TrainData/OversampledAugmentedCOVID-19/COVID-19/"
    for img_path in os.listdir(path):
      #log
      if verbose == 1:
        print(f"[INFO] Opening {path+img_path}")
      img =  image.load_img(path+img_path, target_size=(HEIGHT, WIDTH),color_mode="grayscale")
      img_arr = image.img_to_array(img)
      train_images.append(img_arr)
      train_labels.append(1)
        
    #----------------------------------------------------------------------------------------------------------------------------
    i = 0
    # LOAD AugmentedCOVID in TRAIN DATA
    for img_path in os.listdir(OUTPUT_PATH):
        #log
        if verbose == 1:
            print(f"[INFO] Opening {path+img_path}")
        img =  image.load_img(OUTPUT_PATH+img_path, target_size=(HEIGHT, WIDTH),color_mode="grayscale")
        img_arr = image.img_to_array(img)
        train_images.append(img_arr)
        train_labels.append(1)
        i+=1
    print(f"[INFO] Loaded {i} AugmentedCOVID images")
    x_train = np.array(train_images)
    y_train = np.array(train_labels)
    x_test = np.array(test_images)
    y_test = np.array(test_labels)

    return x_train, y_train, x_test, y_test


## TRANSFER LEARNING
input_shape = (224,224,3)
WIDTH = HEIGHT = 224
FOLDER_LABELS = ["COVID-19","ViralPneumonia"]
DATASET_LABELS = ["NonAugmentedTrain","TrainData","ValData"]
LEARNING_RATES = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
BATCH_SIZE = [5,8,16,32,64,128]
acc_list = []
INPUT_PATH = "/kaggle/input/covid-datasets/datasets/NonAugmentedTrain/COVID-19/"
OUTPUT_PATH = "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/AugmentedCOVID-19/"
verbose = 0

# LOAD DATA
train_images = []
train_labels = []
test_images = []
test_labels = []

# Set batch size for training and validation
batch_size = 32

# List all available models
model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}
accepted_model = ['DenseNet121','DenseNet169','DenseNet201','ResNet101','ResNet101V2','ResNet152','ResNet152V2','ResNet50','ResNet50V2','MobileNet','MobileNetV2','VGG16','VGG19']


# delete pre-trained model not used
for model_name in list(model_dictionary.keys()):
    if model_name not in accepted_model:
        del model_dictionary[model_name]

loadFromFolder("/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/BacterialPneumonia/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/NonAugmentedTrain/BacterialPneumonia/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/ValData/BacterialPneumonia/", 
               0, 500, train_images, train_labels, test_images, test_labels)

loadFromFolder("/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/Normal/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/NonAugmentedTrain/Normal/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/ValData/Normal/", 
               3, 500, train_images, train_labels, test_images, test_labels)
# LOAD DATA
train_images, train_labels,test_images, test_labels = load_images()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(f"[INFO] Test images: {len(test_images)}")
# resize train set
X_train_resized = []
for img in train_images:
  X_train_resized.append(np.resize(img, input_shape) / 255)
X_train_resized = np.array(X_train_resized)
print(f"[INFO] Train set resized: {X_train_resized.shape}")
# resize test set
X_test_resized = []
for img in test_images:
  X_test_resized.append(np.resize(img, input_shape) / 255)
X_test_resized = np.array(X_test_resized)
print(f"Train shape: {X_train_resized.shape}")

model_benchmarks = {'model_name': [], 'num_model_params': [], 'validation_accuracy': [], 'validation_loss': [], 'training_accuracy': [], 'training_loss': []}

num_iterations = int(len(train_images) / batch_size)
for model_name,model in model_dictionary.items():
    pre_trained_model = model(weights='imagenet', include_top=False, input_shape=input_shape)
    headModel = pre_trained_model.output
    headModel = Flatten(name="flatten")(headModel)
    # FC layers
    headModel = Dense(1024, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(4, activation="softmax")(headModel)
    # new headModel on top of baseModel
    model_ = Model(inputs=pre_trained_model.input, outputs=headModel)
    # freeze all CONV layers 
    for layer in pre_trained_model.layers:
        layer.trainable = False
    model_.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1E-6), metrics=["accuracy"])
    print("[INFO] training head...")
    history = model_.fit(X_train_resized, train_labels,batch_size=32,validation_data=(X_test_resized, test_labels),epochs=20)
    scores = model_.evaluate(X_test_resized, test_labels, verbose=2)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # save important values
    model_benchmarks['model_name'].append(model)
    model_benchmarks['num_model_params'].append(pre_trained_model.count_params())
    model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])
    model_benchmarks['validation_loss'].append(history.history['val_loss'][-1])
    model_benchmarks['training_accuracy'].append(history.history['accuracy'][-1])
    model_benchmarks['training_loss'].append(history.history['loss'][-1])

print(model_benchmarks)
benchmark_df = pd.DataFrame(model_benchmarks)
#benchmark_df.sort_values('num_model_params', inplace=True)
# save to external .csv
benchmark_df.to_csv('benchmark_df_lr1E-6.csv', index=False)
benchmark_df
