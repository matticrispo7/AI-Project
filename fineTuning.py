import tensorflow as tf
from keras import callbacks
from tensorflow import keras
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import VGG19,DenseNet121
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import random
import numpy as np


def loadFromFolder(augPath, notAugPath, valPath, classID, n_samples, train_images, train_labels, test_images, test_labels):
    train_aug = random.choices(population=os.listdir(augPath), k=n_samples)
    train_notAug = random.choices(population=os.listdir(notAugPath), k=n_samples)
    ### load AUGMENTED samples in training set
    c = 0
    for img_path in train_aug:
        # LOG
        if verbose == 1:
            print(f"[LOG] loading {augPath+img_path}")
        img =  image.load_img(augPath+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
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
        img =  image.load_img(notAugPath+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
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
        img =  image.load_img(valPath+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
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
            img =  image.load_img(path+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
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
      img =  image.load_img(path+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
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
        img =  image.load_img(OUTPUT_PATH+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
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


def plot_hist(hist):
    plt.plot(hist.history["loss"], label="Train loss")
    plt.plot(hist.history["val_loss"], label="Val loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train loss", "Validation loss"], loc="upper left")
    plt.show()
   
## TRANSFER LEARNING
input_shape = (224,224,3)
WIDTH = HEIGHT = 224
#FOLDER_LABELS = ["BacterialPneumonia", "COVID-19","ViralPneumonia"]
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

# TEST
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
print(f"[INFO]Train set resized: {X_train_resized.shape}")

# resize test set
X_test_resized = []
for img in test_images:
  X_test_resized.append(np.resize(img, input_shape) / 255)
X_test_resized = np.array(X_test_resized)
print(f"[INFO] Test set resized: {X_test_resized.shape}")
print(f"[INFO] Train shape: {train_images.shape}\n[INFO] Test shape: {test_images.shape}")

### DenseNet121 FINE TUNING
# load DenseNet121 model without head
baseModel = DenseNet121(weights='imagenet', include_top = False, input_shape=input_shape)
baseModel.trainable = False
#baseModel.summary()
# new custom head
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
# FC layers
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(4, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
#model.summary()

for layer in baseModel.layers:
	print(f"{layer.name}: {layer.trainable}")
# the base is frozen => train only the head weights
print("[INFO] compiling model...")
opt = Adam(lr=2E-3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the head of the network for a few epochs
print("[INFO] training head...")
# Reduce learning rate when there is a change lesser than <min_delta> in <val_accuracy> for more than <patience> epochs
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

history = model.fit(X_train_resized, train_labels,batch_size=8,validation_data=(X_test_resized, test_labels),epochs=20, callbacks=[learning_rate_reduction])

baseModel.trainable = True
# unfreeze the final set of CONV layers
for layer in baseModel.layers[:-114]:
    layer.trainable = False
# log
for layer in baseModel.layers:
	print(f"{layer.name}: {layer.trainable}")
        
print("[INFO] re-compiling model...")
opt = Adam(lr=2E-5)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])   
history = model.fit(X_train_resized, train_labels,batch_size=16,validation_data=(X_test_resized, test_labels),epochs=10,callbacks=[learning_rate_reduction])
scores = model.evaluate(X_test_resized, test_labels, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
plot_hist(history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.epoch, np.array(history.history['acc']),label='Train accuracy')
plt.plot(history.epoch, np.array(history.history['val_acc']),label = 'Val accuracy')
plt.legend()
plt.show()


### VGG19 FINE TUNING
# load VGG19 model without head
baseModel = VGG19(weights='imagenet', include_top = False, input_shape=input_shape)
baseModel.trainable = False
#baseModel.summary()
# new custom head
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
# FC layers
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(4, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
#model.summary()

for layer in baseModel.layers:
	print(f"{layer.name}: {layer.trainable}")
# the base is frozen => train only the head weights
print("[INFO] compiling model...")
opt = Adam(lr=2E-3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the head of the network for a few epochs
print("[INFO] training head...")
# Reduce learning rate when there is a change lesser than <min_delta> in <val_accuracy> for more than <patience> epochs
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

history = model.fit(X_train_resized, train_labels,batch_size=8,validation_data=(X_test_resized, test_labels),epochs=20, callbacks=[learning_rate_reduction])

baseModel.trainable = True
# unfreeze the final set of CONV layers
for layer in baseModel.layers[:17]:
    layer.trainable = False
# log
for layer in baseModel.layers:
	print(f"{layer.name}: {layer.trainable}")
        
print("[INFO] re-compiling model...")
opt = Adam(lr=2E-5)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])   
history = model.fit(X_train_resized, train_labels,batch_size=16,validation_data=(X_test_resized, test_labels),epochs=10,callbacks=[learning_rate_reduction])
scores = model.evaluate(X_test_resized, test_labels, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
plot_hist(history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.epoch, np.array(history.history['acc']),label='Train accuracy')
plt.plot(history.epoch, np.array(history.history['val_acc']),label = 'Val accuracy')
plt.legend()
plt.show()


