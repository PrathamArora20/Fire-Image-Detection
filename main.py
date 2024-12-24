import os
import numpy as np
from skimage.io import imread 
from skimage.transform import resize 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
import time
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import argparse
from sklearn.tree import DecisionTreeClassifier

def SVCModel():
    ''' This function trains a Support Vector Classifier model on the dataset 
    and prints the accuracy of the model and the time taken to train the model'''


    start = time.time()
    input_files_dir = 'data'
    categories = ["Fire", "Not_Fire"]
    inputData=[]
    imageLabels = []

    for index, imageCategory in enumerate(categories):
        category_path = os.path.join(input_files_dir, imageCategory)
        for file in os.listdir(category_path):
            eachImage = imread(os.path.join(category_path, file), as_gray=True)
            eachImage = resize(eachImage, (32, 32), anti_aliasing=True)
            inputData.append(eachImage.flatten())
            imageLabels.append(index)


    inputData = np.asarray(inputData)
    imageLabels = np.asarray(imageLabels)

    # Split dataset into training, validation, and test sets
    tempX, x_test, tempY, y_test = train_test_split(inputData, imageLabels, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(tempX, tempY, test_size=0.20)

    # Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    C_val = [1,10,100]
    gamma = [0.1,0.01]
    best_c = 0
    best_gamma = 0
    best_accuracy = 0

    for c in C_val:
        for g in gamma:
            SVCModel = SVC(C=c, gamma=g)
            SVCModel.fit(x_train, y_train)
            y_hat_val = SVCModel.predict(x_val)
            accuracy_val = accuracy_score(y_val, y_hat_val)
            print(f"Accuracy for C={c} and gamma={g} is {accuracy_val*100}")
            if accuracy_val > best_accuracy:
                best_accuracy = accuracy_val
                best_c = c
                best_gamma = g
            
    bestSVCModel=SVC(C=best_c, gamma=best_gamma)
    bestSVCModel.fit(x_train, y_train)

    y_hat_test = bestSVCModel.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_hat_test)

    end = time.time()

    print("\nBest Hyperparameters :", best_c, best_gamma)
    print(f"Validation Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Test Accuracy : {accuracy_test * 100:.2f}%")
    print("Time taken to train the model is : ", end-start)

    


def RandomForestModel():
    '''This function trains a Random Forest classifier on the dataset
    and prints the accuracy of the model and the time taken to train the model'''
    
    start = time.time()
    input_files_dir = 'data'
    categories = ["Fire", "Not_Fire"]
    inputData=[]
    imageLabels = []

    for index, imageCategory in enumerate(categories):
        category_path = os.path.join(input_files_dir, imageCategory)
        for file in os.listdir(category_path):
            eachImage = imread(os.path.join(category_path, file), as_gray=True)
            eachImage = resize(eachImage, (32, 32), anti_aliasing=True)
            inputData.append(eachImage.flatten())
            imageLabels.append(index)


    inputData = np.asarray(inputData)
    imageLabels = np.asarray(imageLabels)


    # Split dataset into training, temp, and test sets (60% train, 20% val, 20% test)
    tempX, x_test, tempY, y_test = train_test_split(inputData, imageLabels, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(tempX, tempY, test_size=0.20)


    n_estimators = [25,50,100]
    max_depth = [5, 10]
    best_n = 0
    best_max_depth = 0
    best_accuracy = 0

    for n in n_estimators:
        for d in max_depth:
            RandomForest = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
            RandomForest.fit(x_train, y_train)
            y_hat_val = RandomForest.predict(x_val)
            accuracyValidation = accuracy_score(y_val, y_hat_val)
            print(f"Validation Accuracy for n_estimators={n} and max_depth={d}: {accuracyValidation*100}")
            if accuracyValidation > best_accuracy:
                best_accuracy = accuracyValidation
                best_n = n
                best_max_depth = d

    # Train the model with the best hyperparameters
    BestRandomForestModel = RandomForestClassifier(n_estimators=best_n, max_depth=best_max_depth)
    BestRandomForestModel.fit(x_train, y_train)

    # Evaluate the model on the test set
    y_hat_test = BestRandomForestModel.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_hat_test)

    end = time.time()
    print("\nBest Hyperparameters :", best_n, best_max_depth)
    print(f"Validation Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Test Accuracy : {accuracy_test * 100:.2f}%")
    print("Time taken to train the model is : ", end-start)


def DecisionTreeModel():
    '''This function trains a Decision Tree classifier on the dataset
    and prints the accuracy of the model and the time taken to train the model'''
    
    start = time.time()
    input_files_dir = 'data'
    categories = ["Fire", "Not_Fire"]
    inputData=[]
    imageLabels = []

    # Load and preprocess the images
    for index, imageCategory in enumerate(categories):
        category_path = os.path.join(input_files_dir, imageCategory)
        for file in os.listdir(category_path):
            eachImage = imread(os.path.join(category_path, file), as_gray=True)
            eachImage = resize(eachImage, (32, 32), anti_aliasing=True)
            inputData.append(eachImage.flatten())  # Flatten the image
            imageLabels.append(index)

    inputData = np.asarray(inputData)
    imageLabels = np.asarray(imageLabels)

    # Split dataset into training, validation, and test sets (60% train, 20% val, 20% test)
    tempX, x_test, tempY, y_test = train_test_split(inputData, imageLabels, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(tempX, tempY, test_size=0.20)

    # Hyperparameters for Decision Tree
    max_depth = [5, 10, None]  # Limiting depth to avoid overfitting
    min_samples_split = [2, 10]  # Minimum samples required to split a node
    best_max_depth = 0
    best_min_samples_split = 0
    best_accuracy = 0

    # Hyperparameter tuning through cross-validation (validation set)
    for d in max_depth:
        for s in min_samples_split:
            DecisionTree = DecisionTreeClassifier(max_depth=d, min_samples_split=s, random_state=42)
            DecisionTree.fit(x_train, y_train)
            y_hat_val = DecisionTree.predict(x_val)
            accuracyValidation = accuracy_score(y_val, y_hat_val)
            print(f"Validation Accuracy for max_depth={d} and min_samples_split={s}: {accuracyValidation*100}")
            if accuracyValidation > best_accuracy:
                best_accuracy = accuracyValidation
                best_max_depth = d
                best_min_samples_split = s

    # Train the model with the best hyperparameters
    BestDecisionTreeModel = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split)
    BestDecisionTreeModel.fit(x_train, y_train)

    # Evaluate the model on the test set
    y_hat_test = BestDecisionTreeModel.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_hat_test)

    end = time.time()
    print("\nBest Hyperparameters:", best_max_depth, best_min_samples_split)
    print(f"Validation Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")
    print("Time taken to train the model is:", end - start)


def KNNModel():
    ''' This function trains a K Nearest Neighbors model on the dataset
    and prints the accuracy of the model and the time taken to train the model'''

    start = time.time()

    input_files_dir = 'data'
    categories = ["Fire", "Not_Fire"]

    inputData = []
    imageLabels = []

    for index, imageCategory in enumerate(categories):
        category_path = os.path.join(input_files_dir, imageCategory)
        for file in os.listdir(category_path):
            eachImage = imread(os.path.join(category_path, file), as_gray=True)
            eachImage = resize(eachImage, (32, 32), anti_aliasing=True)
            inputData.append(eachImage.flatten())
            imageLabels.append(index)



    inputData = np.asarray(inputData)
    imageLabels = np.asarray(imageLabels)

    # Split dataset into training, temp, and test sets (60% train, 20% val, 20% test)a
    tempX, x_test, tempY, y_test = train_test_split(inputData, imageLabels, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(tempX, tempY, test_size=0.20)

    # Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)


    n_neighbors = [10, 20, 50 ]
    metric = ['euclidean', 'manhattan']
    best_n_neighbors = 0
    best_metric = ''
    best_accuracy = 0

    for n in n_neighbors:
        for m in metric:
            KNNModel = KNeighborsClassifier(n_neighbors=n, metric=m)
            KNNModel.fit(x_train, y_train)
            y_hat_val = KNNModel.predict(x_val)
            accuracyValidation = accuracy_score(y_val, y_hat_val)
            print(f"Validation Accuracy for n_neighbors={n} and metric={m}: {accuracyValidation * 100}%")
            if accuracyValidation > best_accuracy:
                best_accuracy = accuracyValidation
                best_n_neighbors = n
                best_metric = m

    KNNBestModel = KNeighborsClassifier(n_neighbors=best_n_neighbors, metric=best_metric)
    KNNBestModel.fit(x_train, y_train)

    y_hat_test = KNNBestModel.predict(x_test) 
    accuracyTest = accuracy_score(y_test, y_hat_test) 

    end = time.time()

    print("\nBest Hyperparameters :", best_metric, best_n_neighbors)
    print(f"Validation Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Test Accuracy : {accuracyTest * 100:.2f}%")
    print("Time taken to train the model is : ", end-start)



def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--model", help="Choose the model you want to run", required=True)
    args = argparser.parse_args()

    if args.model == "SVC":
        SVCModel()
    elif args.model == "RandomForest":
        RandomForestModel()
    elif args.model == "KNN":
        KNNModel()
    
# if __name__ == "__main__":
#     main()

DecisionTreeModel()