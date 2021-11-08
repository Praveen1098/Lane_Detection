# Lane Detection
- In this project we accomplish two tasks,
  - Enhance the contrast and improve the visual appearance of a night time driving video sequence.
  - We implement an algorithm to detect lanes on the road as well as estimate the road curvature to predict car turns.

![Lane Prediction](/Output/Sample.gif)


## Dependencies

- Python 3.X
- NumPy
- OpenCV

## Usage

- Run the python scripts in the current directory which contains all the code.
  - laneDetection_input_1.py - Lane detection script for Input_1
  - laneDetection_input_2.py - Lane detection script for Input_2
  - videoQualityEnhancer.py  - Night time driving video enhancement for Input_3

- Place the relative path of the video you want to test for, 
```
cap = cv2.VideoCapture('Data/Input_1.mp4').
```
- Open terminal and type the following commands for different input videos,

  - Input_1
  
    ```
    python3 laneDetection_input_1.py
    ```
  - Input_2
  
    ```
    python3 laneDetection_input_2.py
    ```
  - Input_3
  
    ```
    python3 videoQualityEnhancer.py
    ```


