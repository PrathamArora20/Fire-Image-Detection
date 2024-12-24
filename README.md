# Fire-Image-Detection
Image Recognition that recognizes fire. Uses Random Forest , SVM and KNN Models


Introduction 


In today's day and age, safety is one of the most important concerns. If there is an accident or an incident which can cause a safety issue to the public, it is crucial to call the necessary authorities quickly to deal with the situation as soon as possible. One of these incidents include fires. Fire related accidents is one of the most scary and life-threatening incidents. Whenever there is a fire, the fire department needs to be informed quickly to come deal with it. A lot of places are equipped with smoke detectors to detect a fire and immediately call the fire department. This helps in deescalating the situation quickly. 


But there are situations where fire detectors are not useful. It could be a fire in a public place like a garden where fire detectors cannot be used or an accident on the highway, and the fire department needs to be informed immediately. This is where this project comes into the picture. This project is a binary image classification project that detects whether or not there is a fire. Using the help of cameras in public places, this model will use the pictures from the cameras and detect if there is a fire or not. (There could be another model that converts videos to constant images, but we would not be covering that in this project). This project implements machine learning models to learn and predict on the image if there is a fire. If the model believes there is a fire, then the fire department would be called immediately to handle the situation. 

This project implements 3 different types of image classification models to classify the image to either fire or not fire. 

Support Vector Classifier (SVC)
Random Forest Classifier  
K Nearest Neighbors (KNN)


Problem Formulation

For the problem stated above, the input would be images of random places. These places would either have a fire or not. 



For example : <br>
![Img_20897](https://github.com/user-attachments/assets/2e63c473-3b9c-445b-8adf-28a3ecad9957) <br>
This image has fire <br>

![Img_32437](https://github.com/user-attachments/assets/a952551e-fd71-4449-95bc-d62e5def3975) <br>
This image has no fire <br>


