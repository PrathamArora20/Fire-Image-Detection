# Fire-Image-Detection
Image Recognition that recognizes fire. Uses Random Forest , SVM and KNN Models


## Introduction 


In today's day and age, safety is one of the most important concerns. If there is an accident or an incident which can cause a safety issue to the public, it is crucial to call the necessary authorities quickly to deal with the situation as soon as possible. One of these incidents include fires. Fire related accidents is one of the most scary and life-threatening incidents. Whenever there is a fire, the fire department needs to be informed quickly to come deal with it. A lot of places are equipped with smoke detectors to detect a fire and immediately call the fire department. This helps in deescalating the situation quickly. 


But there are situations where fire detectors are not useful. It could be a fire in a public place like a garden where fire detectors cannot be used or an accident on the highway, and the fire department needs to be informed immediately. This is where this project comes into the picture. This project is a binary image classification project that detects whether or not there is a fire. Using the help of cameras in public places, this model will use the pictures from the cameras and detect if there is a fire or not. (There could be another model that converts videos to constant images, but we would not be covering that in this project). This project implements machine learning models to learn and predict on the image if there is a fire. If the model believes there is a fire, then the fire department would be called immediately to handle the situation. 

This project implements 3 different types of image classification models to classify the image to either fire or not fire. 

Support Vector Classifier (SVC)
Random Forest Classifier  
K Nearest Neighbors (KNN)


## Problem Formulation

For the problem stated above, the input would be images of random places. These places would either have a fire or not. 



For example : <br>
![Img_20897](https://github.com/user-attachments/assets/2e63c473-3b9c-445b-8adf-28a3ecad9957) <br>
This image has fire <br>

![Img_32437](https://github.com/user-attachments/assets/a952551e-fd71-4449-95bc-d62e5def3975) <br>
This image has no fire <br>

The dataset I found was from here:

https://www.kaggle.com/datasets/diversisai/fire-segmentation-image-dataset?select=Segmentation_Mask

From this dataset, I only used the images in the folder Fire and Not_fire. This dataset has images of both places with fire and not a fire. There are a total of 9,807 images in the dataset (3,196 not fire and 6,611 fire). (I used half of the original data only as the original data was significantly more, and it took a lot of time for training and required a high computational cost like higher spec machine for training which I did not have)  So this would be a binary image classification model that trains the model to recognize if there is a fire or not (classification between 2 labels). These 9,807 images would act like an input to the model where 60% of the data would be for training, 20 percent of the data would be for validation to see how well the data performs on unseen images and the rest 20 percent of the data would be for test set to see the model performance. 

The expected out from the models would be a classification to either tell if the input image has a fire or not. The expected output for all 3 models would be the same. It predicts an output which is a predicted class label. The predicted class label is either 0 or 1. If the predicted class label is 0 then it means “Fire” and if the predicted class label is 1 then it means “Not Fire”.

## Approaches and Baseline 

So for all the 3 models, I first read the data. Then, the data was loaded in a 32 x 32 picture which was then turned into a grey image to a 1 dimensional vector. The data was then split into training, validation and testing data sets.(60 20 20 split)
 
Then there is an additional normalization step for SVC and KNN. We normalize the data for KNN and SVC because both algorithms are distance based algorithms , so the distance between the points in the feature space plays a big role here. Normalization makes sure that every feature in the data has a zero mean , so that the distance calculation is not weighted towards some images. Normalization makes sure that all the features contribute equally in the prediction 

For KNN , the distance calculation is performed by calculating the distance between points to the nearest neighbours. If there are features with higher pixel values or larger ranges, those features could have a higher weightage in the distance calculation. This is troublesome, as then some images or features might hold more weightage than others, which disturbs the balance. When we normalize the pixels of the images, this makes sure that the distance calculation is done fairly.

For SVC, It is also a distance based algorithm. For SVC, the model calculates distance between points in the high dimensional space, so if there are features which are larger compared to others , this would affect the distance calculation and the prediction will be weighted more towards the features with large images. 

We don't need normalization for Random Forest classification here because the model is not sensitive to scale of features and is not a distance based mode. Random Forest classification works by have a decision tree on features. Hence , This problem doesn't affect Random Forest Classification.

Going more in depth in the 3 machine learning image classification models that were implemented:

### Support Vector Classifier (SVC) 

SVC is a model that finds the best hyperplane that would try to separate the different classes while also trying to maximize the margins between them. 

Implementation: After the reading and normalization of the data was done , an SVC model was created with the Hyperparameters like Regularization parameter C as either 1 ,10 or 100 and Kernel Coefficient γ as either 0.01 or 0.1.  With the help of hyperparameter tuning, we choose the hyperparameters which show the best result in the validation set. I compare the validation set to pick the hyperparameters because it is not used in training and gives a rough estimate of how the model works on unseen data (How well the data generalizes)


### K-Nearest Neighbors ( KNN) 

KNN is a model that classifies the data points and images based on the neighbours. It sees the neighbours and depending on most of the classification done in the neighbourhood , it will classify the data point 

Implementation: After the reading and normalization of the data was done , A KNN model was created with the hyperparameter like k value, which is how many neighbours are considered when assigning a classification. In this problem, we have chosen k to be either 10 , 20 or 50 and the distance metric to either be Euclidean or Manhattan.  With the help of hyperparameter tuning, we choose the hyperparameters which show the best result in the validation set


### Random Forest 

Random Forest is a model that makes multiple different decision trees on randomly selected sets of the data. Then it combines their prediction to make the accuracy better.

Implementation: After the reading and normalization of the data was done, A Random Forest model was created with hyperparameter like n_estimator that controls the number of trees , ( More trees) and max_depth which determines the maximum depth of a tree. In this problem, we have chosen  n_estimator to be either 25, 50 or 100 and max_depth to be 5 or 10. With the help of hyperparameter tuning, we choose the hyperparameters which show the best result in the validation set

## Hyper Parameter Tuning 

We tune Hyperparameters for the models and choose the best one which optimizes the results and pick the ones that have the best Validation Accuracy. The below values have been picked as they performed better than the other values that were tested , and these values gave more promising results.


### SVC

The parameters we tune are regularization parameter C and Kernel Coefficient γ (gamma) 




