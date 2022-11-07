from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import random
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator

WIDTH = 256
HEIGHT = 256
FOLDER_LABELS = ["COVID-19","ViralPneumonia"]
DATASET_LABELS = ["NonAugmentedTrain","TrainData","ValData"]
LEARNING_RATES = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
BATCH_SIZE = [5,8,16,32,64,128]
acc_list = []
INPUT_PATH = "/kaggle/input/covid-datasets/datasets/NonAugmentedTrain/COVID-19/"
OUTPUT_PATH = "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/AugmentedCOVID-19/"
verbose = 0

# total images in training set for each class
totalBacterial = 0
totalNormal = 0
totalCovid = 0
totalViral = 0

def fit_model(train_images, train_labels, test_images, test_labels, lr, batch, epochs):
    network = models.Sequential()
    network.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(WIDTH,HEIGHT,1)))
    network.add(layers.MaxPooling2D((2,2)))
    network.add(layers.Conv2D(32,(3,3),activation="relu"))
    network.add(layers.MaxPooling2D((2,2)))
    network.add(layers.Conv2D(64,(3,3),activation="relu"))
    network.add(layers.MaxPooling2D((2,2)))
    network.add(layers.Conv2D(64,(3,3),activation="relu"))
    network.add(layers.MaxPooling2D((2,2)))
    network.add(layers.Conv2D(128,(3,3),activation="relu"))
    network.add(layers.MaxPooling2D((2,2)))
    network.add(layers.Conv2D(128,(3,3),activation="relu"))
    network.add(layers.MaxPooling2D((2,2)))
    network.add(layers.Flatten())
    network.add(layers.Dense(512,activation="relu"))
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(4,activation="softmax"))
    network.summary()
    
    network.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(learning_rate=lr),metrics=['acc'])
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
    history = network.fit(train_images, train_labels, epochs=epochs, batch_size=batch, validation_data=(test_images, test_labels), callbacks=[learning_rate_reduction])

    test_loss,test_acc = network.evaluate(test_images,test_labels)
    print(f"Test accuracy with lr {lr}: {test_acc}\nTest loss with lr {lr}:{test_loss}")
    acc_list.append(test_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history.epoch, np.array(history.history['loss']),label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),label = 'Val loss')
    plt.legend()
    plt.show()
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(history.epoch, np.array(history.history['acc']),label='Train accuracy')
    plt.plot(history.epoch, np.array(history.history['val_acc']),label = 'Val accuracy')
    plt.legend()
    plt.show()
    
    
def gridSearchBatchSize(train_images, train_labels, test_images, test_labels, epochs):
    for b in BATCH_SIZE:
        print(f"[INFO] Training the model with batch {b}")
        fit_model(train_images, train_labels, test_images, test_labels, 1E-5, b, epochs)    
    # log
    for i in range(len(acc_list)):
        print(f"[RESULT] Batch_Size: {BATCH_SIZE[i]} -> acc: {acc_list[i]}")
    
def grid_search_lr(train_images, train_labels, test_images, test_labels, batch, epochs):
    for lr in LEARNING_RATES:
        print(f"Training the model with lr {lr}")
        fit_model(train_images, train_labels, test_images, test_labels, lr, batch, epochs)    
    # log
    for i in range(len(acc_list)):
        print(f"[RESULT] lr: {LEARNING_RATES[i]} -> acc: {acc_list[i]}")

def generateAugmentedImages(input_path, output_path):
    for img_path in os.listdir(INPUT_PATH):
        img = image.load_img(INPUT_PATH+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
        print(f"IMG: {img_path}")
        # convert to numpy array
        img_arr = image.img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(img_arr, 0)
        data_generator = ImageDataGenerator(
            brightness_range=[0.5,1.0], 
            rotation_range=40, 
            horizontal_flip=True, 
            zoom_range=0.2, 
            shear_range=0.2, 
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest')
        it = data_generator.flow(samples, batch_size=1,save_to_dir=OUTPUT_PATH, save_format="jpg")
        # 15 transformation for each image
        for i in range(15):
            batch = it.next()


def loadFromFolder(augPath, notAugPath, valPath, classID, n_samples, train_images, train_labels, test_images, test_labels):
    trainingImagesLoaded = 0
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
        trainingImagesLoaded += 1
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
        trainingImagesLoaded += 1
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
    return trainingImagesLoaded
                  
                  
def load_images(totCovid, totViral):
    for i in range(len(DATASET_LABELS)):  
        c=0
        for j in range(len(FOLDER_LABELS)):
          path ="/kaggle/input/d/mattiacrispino/covid-datasets/datasets/"+DATASET_LABELS[i]+"/"+FOLDER_LABELS[j]+"/" 
          for img_path in os.listdir(path):
            img =  image.load_img(path+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
            img_arr = image.img_to_array(img)
            # log
            if verbose == 1:
              print(f"[INFO] Opening {path+img_path}")
            # train data
            if i != 2:
              if j == 0:
                totCovid += 1
              else:
                totViral +=1
              train_images.append(img_arr)
              train_labels.append(j+1)
              c+=1
            # test data
            else:  
              test_images.append(img_arr)
              test_labels.append(j+1)
        print(f"[INFO] Loaded {c} images")
   

    # LOAD OVERSAMPLED COVID-19
    i=0
    path = "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/OversampledAugmentedCOVID-19/COVID-19/"
    for img_path in os.listdir(path):
      #log
      if verbose == 1:
        print(f"[INFO] Opening {path+img_path}")
      img =  image.load_img(path+img_path, target_size=(HEIGHT, WIDTH), color_mode="grayscale")
      img_arr = image.img_to_array(img)
      train_images.append(img_arr)
      train_labels.append(1)
      totCovid += 1
      i+=1
    print(f"[INFO] Loaded {i} OversampledAugmentedCOVID-19 images")
        
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
        totCovid += 1
        i+=1
    print(f"[INFO] Loaded {i} AugmentedCOVID images")

    x_train = np.array(train_images)
    y_train = np.array(train_labels)
    x_test = np.array(test_images)
    y_test = np.array(test_labels)

    return x_train, y_train, x_test, y_test, totCovid, totViral

# LOAD DATA
train_images = []
train_labels = []
test_images = []
test_labels = []



# load BacterialPneumonia
totalBacterial = loadFromFolder("/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/BacterialPneumonia/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/NonAugmentedTrain/BacterialPneumonia/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/ValData/BacterialPneumonia/", 
               0, 500, train_images, train_labels, test_images, test_labels)
# load Normal
totalNormal = loadFromFolder("/kaggle/input/d/mattiacrispino/covid-datasets/datasets/TrainData/Normal/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/NonAugmentedTrain/Normal/", 
               "/kaggle/input/d/mattiacrispino/covid-datasets/datasets/ValData/Normal/", 
               3, 500, train_images, train_labels, test_images, test_labels)

# LOAD DATA
train_images, train_labels,test_images, test_labels, totalCovid, totalViral = load_images(totalCovid, totalViral)


print(f"[INFO] Train images: {len(train_images)}")
print(f"\n[INFO] Total Bacterial: {totalBacterial}\n[INFO] Total Normal: {totalNormal}\n[INFO] Total COVID: {totalCovid}\n[INFO] Total Viral: {totalViral}\n")

# RESHAPE
train_images = train_images.reshape((len(train_images), WIDTH, HEIGHT, 1))
train_images = train_images.astype('float32') / 255 
test_images = test_images.reshape((len(test_images), WIDTH, HEIGHT, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#grid_search_lr(train_images, train_labels, test_images, test_labels, 32, 50)
#gridSearchBatchSize(train_images, train_labels, test_images, test_labels, 50)
fit_model(train_images, train_labels, test_images, test_labels, 1E-5, 8, 150)