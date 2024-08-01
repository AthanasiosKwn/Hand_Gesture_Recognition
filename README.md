# Hand_Gesture_Recognition
Hand Gesture Recognition Using Motion Energy Images - Dynamic Time Warping (DTW).


![image](https://github.com/user-attachments/assets/024c5467-3cdd-4702-a88e-530fd4879a21)



The goal of this project is the visual recognition of four different gestures ('Click', 'No', 
'Rotate', 'StopGraspOk'). First, gesture recognition is executed using a method based on motion energy images (MEI's). 
The second method uses 2D Dynamic Time Warping (DTW) to match vector representations of gestures (two different representantions are tested).
In both cases a K-NN classifier is used to classify the test images. 

This repository contains a dropbox link (https://www.dropbox.com/scl/fi/owshkn52m8z4muo88hrkx/img_database.zip?rlkey=ltu0kcfl7ocbf0z08l6jm3fqx&st=d8zi3f1j&dl=0) to the database of image sequences (there are 4 sub-folders one for each kind of gesture, not split into train - test set), the .py code file and a .pdf file with the detailed project report.
The code is written in Python utilizing functions from the following modules: OpenCV, Matplotlib, sklearn, pandas, tslearn and NumPy.
