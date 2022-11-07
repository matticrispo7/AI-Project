# CNN for classification of COVID-19 pneumonia
The objective of the project is to build a CNN model with subsequent training phase, validation and comparison of the results with some pretrained network models on which fine tuning is performed.

## 
1. Bacterial Pneumonia
2. COVID-19
3. Normal
4. Viral Pneumonia
![classesHorizontal](https://user-images.githubusercontent.com/22591922/200346692-b36b14b0-706b-470f-b4b8-e66fbab93094.png)

## Dataset
The starting dataset and the training set can be found [here](https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset).\\
To balance the dataset, 1000 samples per class were chosen.\\
Since some classes had fewer samples than required, the data augmentation technique with 15 transformations per sample was applied.\\
An example of the application of this technique on a sample of COVID-19 is shown below.
<p align="center">
<img height="200" src="https://user-images.githubusercontent.com/22591922/200353069-22301420-553c-4934-b0d0-c5539136d5fa.png">
</p>
## CNN structure
The CNN is composed by several *convolutional* layers followed by *pooling* ones.
Then, the feature map is flattened and two *Dense* layers composed by 512 and 4 respectively are added with a *Dropout* layer in between.

![CNN](https://user-images.githubusercontent.com/22591922/200343426-da4b05b1-4f5d-4d6f-9ea3-882445e67511.png)
