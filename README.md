# CNN for classification of COVID-19 pneumonia
The objective of the project is to build a CNN model with subsequent training phase, validation and comparison of the results with some pretrained network models on which fine tuning is performed.
The dataset and training set can be found [here] (https://www.kaggle.com/darshan1504/covid19-detection-xray-dataset)

## Classes 


## CNN structure
The CNN is composed by several *convolutional* layers followed by *pooling* ones.
Then, the feature map is flattened and two *Dense* layers composed by 512 and 4 respectively are added with a *Dropout* layer in between.

![CNN](https://user-images.githubusercontent.com/22591922/200343426-da4b05b1-4f5d-4d6f-9ea3-882445e67511.png)
