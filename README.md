# CNN for classification of COVID-19 pneumonia
This is a master's degree project of Artificial Intelligence developed in 2021.

# Goal
The objective of the project is to build a CNN model with subsequent training phase, validation and comparison of the results with some pretrained network models on which fine tuning is performed.<br />
More details can be found in the documentation provided.

## Classes to detect
The classes the network has to detect are the following:
1. Bacterial Pneumonia
2. COVID-19
3. Normal
4. Viral Pneumonia <br />
An example of these (respectively from left to right) are provided in the figure below.
<p align="center">
<img height="100" src="https://user-images.githubusercontent.com/22591922/200346692-b36b14b0-706b-470f-b4b8-e66fbab93094.png">
</p>

## Dataset
The starting dataset and the training set can be found [here](https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset).<br />
To balance the dataset, 1000 samples per class were chosen.<br />
Since some classes had fewer samples than required, the data augmentation technique with 15 transformations per sample was applied.<br />
An example of the application of this technique on a sample of COVID-19 is shown below.
<p align="center">
<img height="200" src="https://user-images.githubusercontent.com/22591922/200353069-22301420-553c-4934-b0d0-c5539136d5fa.png">
</p>

## CNN structure
The CNN is composed by several *convolutional* layers followed by *pooling* ones.
Then, the feature map is flattened and two *Dense* layers composed by 512 and 4 respectively are added with a *Dropout* layer in between.<br />
The visual representation of the network's structure is the following:
<p align="center">
<img height="300" src="https://user-images.githubusercontent.com/22591922/200343426-da4b05b1-4f5d-4d6f-9ea3-882445e67511.png">
</p>


## Results
The final accuracy obtained is 78,14%. Accuracy and loss graphs are shown below.
<p align="center">
<img height="200" src="https://user-images.githubusercontent.com/22591922/200357030-a436d073-56c3-4e4a-86bd-afdb58802365.png">
</p>
