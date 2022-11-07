# CNN for classification of COVID-19 pneumonia
The objective of the project is to build a CNN model with subsequent training phase, validation and comparison of the results with some pretrained network models on which fine tuning is performed.

## 
1. Bacterial Pneumonia
2. COVID-19
3. Normal
4. Viral Pneumonia
![classesHorizontal](https://user-images.githubusercontent.com/22591922/200346692-b36b14b0-706b-470f-b4b8-e66fbab93094.png)

## Dataset
The starting dataset and the training set can be found [here](https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset).
To balance the dataset, 1000 samples per class were chosen.
Since some classes had fewer samples than required, the data augmentation technique with 15 transformations per sample was applied.
An example of the application of this technique on a sample of COVID-19 is shown below.
![originalSample](https://user-images.githubusercontent.com/22591922/200350097-6cd7b3c3-c2d3-411b-82f2-4b411bf3b962.png)*original sample*

![alt-text-1](https://user-images.githubusercontent.com/22591922/200350097-6cd7b3c3-c2d3-411b-82f2-4b411bf3b962.png "Original sample") ![alt-text-2](i[mage2.png](https://user-images.githubusercontent.com/22591922/200350641-45a933d2-a9b9-41f3-a019-0d36390d1bdd.png) "Augmented samples")
## CNN structure
The CNN is composed by several *convolutional* layers followed by *pooling* ones.
Then, the feature map is flattened and two *Dense* layers composed by 512 and 4 respectively are added with a *Dropout* layer in between.

![CNN](https://user-images.githubusercontent.com/22591922/200343426-da4b05b1-4f5d-4d6f-9ea3-882445e67511.png)
