# ENPM-673-Project-2
The project focuses on enhancing the contrast and improve the visual appearance of the night video sequence and perform simple Lane Detection to mimic Lane Departure Warning systems used in Self Driving Cars.Our
task will be to design an algorithm to detect lanes on the road, as well as estimate the road curvature to predict car turns.

![Lane Prediction](/Output/Sample.gif)


## Packages Required

- NumPy

- OpenCV

## To run the codes

- Run the python files in the current directory which contains all the codes.

- The code runs with the sample video "nightvideo.mp4" placed in the Data Folder for Problem 1 and for Lane detection,the code runs for "challenge_video.mp4" and "projectVideo.avi" for data_1.

- Place the relative path of the video you want to run in,cap = cv2.VideoCapture('Data/nightvideo.mp4').

- Open terminal and type the following commands for respective solutions 
Enhanced night time driving Video 
```
python3 Problem1.py
```
Dataset1
```
python3 laneDetection_data1.py
```
Challenge Video
```
python3 laneDetection_data2.py
```


