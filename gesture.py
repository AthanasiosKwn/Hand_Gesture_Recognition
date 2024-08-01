import cv2 as cv
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from collections import Counter

# Recognition through Motion Energy Images

def motion_energy_image_generator(image_files, folder, sequence):
    motion_energy_img = np.zeros_like(cv.imread(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}\{image_files[0]}', cv.IMREAD_GRAYSCALE))
    # Loop through the image sequence. For a list image_files of len(image_files) number of files, the index ranges from 0 to len(image_files) - 1. 
    # That means that the loop must break when the index is lean(image_files)-2 to avoid list index out of range error
    for i in range(len(image_files)):
        if i == (len(image_files)-2):
            break
        else:
            imagei = cv.imread(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}\{image_files[i]}', cv.IMREAD_GRAYSCALE)
            imagei_plus1 = cv.imread(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}\{image_files[i+1]}', cv.IMREAD_GRAYSCALE)
            # Absolute Difference between two consecutive images - frames
            differene_image = cv.absdiff(imagei,imagei_plus1)
            # Apply a threshold to the difference image
            _, motion = cv.threshold(differene_image, 50, 1, cv.THRESH_BINARY)
            # Motion Energy Image
            motion_energy_img += motion

    # Normalize the MEI to the range [0, 255] for display
    motion_energy_img = cv.normalize(motion_energy_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return motion_energy_img
    


# X is the dataset used to train the KNN classifier. y contains the labels. 
# The dataset is built in such a way that each row is a flattened motion energy image
X = []
y = []

# The four folders containing the different sequences for the four different hand gestures
gestures_folders = os.listdir(r'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel')
for folder in gestures_folders:
    # Sequences of a specific hand gesture
    sequences = os.listdir(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}')
    for sequence in sequences: 
        image_files = os.listdir(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}')
        image_files.sort()
        motion_energy_image = motion_energy_image_generator(image_files,folder,sequence)
        cv.imwrite(f'MEI_{folder}_{sequence}.jpg',motion_energy_image)
        flattened_mei = motion_energy_image.flatten().tolist()
        X.append(flattened_mei)
        y.append(folder)

# Convert dataset and labels lists to pd dataframes
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)




# Split dataset to train and test set. Train set will consist of the first 5 MEI of each gesture

X_train = X_df.loc[[0,1,2,3,4,15,16,17,18,19,29,30,31,32,33,42,43,44,45,46]]
X_test = X_df.drop([0,1,2,3,4,15,16,17,18,19,29,30,31,32,33,42,43,44,45,46])
# Convert to np arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_df.loc[[0,1,2,3,4,15,16,17,18,19,29,30,31,32,33,42,43,44,45,46]]
y_test = y_df.drop([0,1,2,3,4,15,16,17,18,19,29,30,31,32,33,42,43,44,45,46])

# Convert to np arrays and flatten
y_train = y_train.to_numpy().flatten() 
y_test = y_test.to_numpy().flatten()

# Train K-NN classifier and produce the confusion matrix for different values of k
for neighbors in range(2,6):
    # The model
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    # Fit the model
    knn.fit(X_train,y_train)
    # Make predictions on the test set
    y_predictions = knn.predict(X_test)
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_predictions)
    # Calculcate the accuracy of classification
    accuracy = accuracy_score(y_test, y_predictions)
    print(f"Accuracy for k = {neighbors}: {accuracy} ",)
    # Save the confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=knn.classes_).plot()
    plt.savefig(f'confusion_matrix_{neighbors}.png')




# Recognition through DTW

# Create a function that loops through the images of a sequence 
# and produces the vector containing the positions of the hand during the gesture.

def gesture_vector_generator(image_files, folder, sequence):
    '''Receives the image files of a sequence of images - gesture and computes a vector describing the gesture.
       This vector contains a list of average positions (x_avg,y_avg) of the hand at each image-frame of the sequence.
       x_avg is the average x position of all the contour points, y_avg is the average y position of all the contour points
    '''
    # 'Positions' is the vector that describes the gesture for a specific sequence of images. It contains the positions of the hand
    # (x_avg, y_avg) during this gesture - sequence of images
    positions = []

    for image_file in image_files:

        # Read the image
        image = cv.imread(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}\{image_file}', cv.IMREAD_GRAYSCALE)
        
        # Find the edges  
        edges = cv.Canny(image,30,90)

        # Find all external contours (no hierarchical connection)
        contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Iterate through the points of every contour to find the avg x and the avg y position. Point (x_avg,y_avg) represents hand position
        x=0
        y=0
        total_points=0
        for contour in contours:
            for point in contour:
                x += point[0][0]
                y += point[0][1]
                total_points += 1
        x_avg = x // total_points
        y_avg = y// total_points
        positions.append([x_avg, y_avg])

    return positions
        
# X contains the vectors returned by gesture_vector_generator
# y contains the labels
X = []
y = []

# The four folders containing the different sequences for the four different hand gestures
gestures_folders = os.listdir(r'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel')
for folder in gestures_folders:
    # Sequences of a specific hand gesture
    sequences = os.listdir(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}')
    for sequence in sequences: 
        image_files = os.listdir(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}')
        image_files.sort()
        position_vector = gesture_vector_generator(image_files,folder,sequence)
        X.append(position_vector)
        y.append(folder)



# Initialize datasets 
X_train = []
X_test = []
y_train = []
y_test = []

# Split datasets
train_indices = [0,1,2,3,4,15,16,17,18,19,29,30,31,32,33,42,43,44,45,46]
X_train = [X[i] for i in train_indices]
X_test = [X[i] for i in range(len(y)) if i not in train_indices]

y_train = [y[i] for i in train_indices]
y_test = [y[i] for i in range(len(y)) if i not in train_indices]


# For each query gesture in the X_test dataset, we compute the distance from each labeled gesture in the X_train dataset using DTW.
# We find the k nearest neighbors for it and its class is determined by the majority vote. If there is a tie, the selected label is the first in order of appearance.

# Classifier class
class Classifier:
    def __init__(self,num_neighbors):
        self.num_neighbors = num_neighbors
    
    # Fit and predict method. 
    def fit_predict(self, X_train, X_test, y_train):
        # Calculate predictions
        y_predictions = []
        for gesture in X_test:
            distances = []
            for labeled_gesture in X_train:
                # Calculate DTW distance
                dtw_distance = dtw(np.array(gesture), np.array(labeled_gesture))
                distances.append(dtw_distance)

            # Sort indices in ascending order based on distance values
            sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])

            # Get the indices of the 3 closest neighbors
            lowest_indices = sorted_indices[:self.num_neighbors]

            votes = [y_train[i] for i in lowest_indices]
            # Create a Counter object
            counter = Counter(votes)

            # Find the most common element - majority vote
            majority_vote,_ = counter.most_common(1)[0]
            y_predictions.append(majority_vote)
        return y_predictions

# Fit and predict. Compute and save the confusion matrix
for k in range(2,6):
    classifier = Classifier(num_neighbors=k)
    y_predictions = classifier.fit_predict(X_train, X_test, y_train)
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_predictions)
    # Calculcate the accuracy of classification
    accuracy = accuracy_score(y_test, y_predictions)
    print(f"Accuracy for k = {k}: {accuracy} ")
    # Save the confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=knn.classes_).plot()
    plt.savefig(f'confusion_matrix_dtw_{k}.png')





# DTW recognition - extended vector representation of gestures

def gesture_vector_extended_generator(image_files, folder, sequence):
    '''Receives the image files of a sequence of images - gesture, and computes a vector describing the gesture.
       This vector contains a list of representantions (x_avg,y_avg,w,h) of the hand at each image-frame of the sequence.
       x_avg is the average x position of all the contour points, y_avg is the average y position of all the contour points
       w is the width, and h is the height of the minimum bounding rectangle containing the hand.
    '''
    # Positions is the vector that describes the gesture for a specific sequence of images. 
    positions = []

    for image_file in image_files:

        # Read the image
        image = cv.imread(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}\{image_file}', cv.IMREAD_GRAYSCALE)
        
        # Find the edges  
        edges = cv.Canny(image,30,90)

        # Find all external contours (no hierarchical connection)
        contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Iterate through the points of every contour to find the avg x and the avg y position. 
        x=0
        y=0
        total_points=0
        for contour in contours:
            for point in contour:
                x += point[0][0]
                y += point[0][1]
                total_points += 1
        x_avg = x // total_points
        y_avg = y// total_points
        
        # Find external contour of maximum area, fit the MBR and obtain the width and height of it. Point (x_avg,y_avg,w,h) represents the hand.
        contours_ext,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_contour  = max(contours_ext, key=cv.contourArea)
        _, _, w, h = cv.boundingRect(contour)
        positions.append([x_avg, y_avg, w, h])


    return positions


X = []
y = []

# The four folders containing the different sequences for the four different hand gestures
gestures_folders = os.listdir(r'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel')
for folder in gestures_folders:
    # Sequences of a specific hand gesture
    sequences = os.listdir(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}')
    for sequence in sequences: 
        image_files = os.listdir(fr'C:\Users\xthan\Desktop\masters\courses\video processing\project3\dhp_marcel\{folder}\{sequence}')
        image_files.sort()
        position_vector = gesture_vector_extended_generator(image_files,folder,sequence)
        # X contains the vectors returned by gesture_vector_generator
        # y contains the labels
        X.append(position_vector)
        y.append(folder)

#Initialize datasets    
X_train = []
X_test = []
y_train = []
y_test = []

# Split into test and train set
train_indices = [0,1,2,3,4,15,16,17,18,19,29,30,31,32,33,42,43,44,45,46]
X_train = [X[i] for i in train_indices]
X_test = [X[i] for i in range(len(y)) if i not in train_indices]

y_train = [y[i] for i in train_indices]
y_test = [y[i] for i in range(len(y)) if i not in train_indices]


# For each query gesture in the X_test dataset, we compute each distance from each labeled gesture in the X_train dataset using DTW.
# We find the k nearest neighbors for it and its class is determined by the majority vote. If there is a tie, the selected label is the first in order of appearance.

# Fit and predict. Compute and save the confusion matrix 
for k in range(2,6):
    classifier = Classifier(num_neighbors=k)
    y_predictions = classifier.fit_predict(X_train, X_test, y_train)
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_predictions)
    # Calculcate the accuracy of classification
    accuracy = accuracy_score(y_test, y_predictions)
    print(f"Accuracy for k = {k}: {accuracy} ")
    # Save the confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=knn.classes_).plot()
    plt.savefig(f'confusion_matrix_dtw_extended_{k}.png')

