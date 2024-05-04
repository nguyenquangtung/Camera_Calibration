# Camera Calibration

Created on Sunday, April 21, 2024  
Author: Tung Nguyen - Handsome  
Reference: Camera Calibration by OpenCV ([OpenCV Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html))

![Banner](https://github.com/nguyenquangtung/Camera_Calibration/assets/59195029/2ad0cdd1-dd48-4083-9766-8d42533cd9a0)

## Overview

This Python script provides a class, `CameraCalibration`, to perform camera calibration based on a set of image points and their corresponding object points / undistort image or frame from camera

## Description

This script includes functionalities to:

- Calculate camera calibration data based on a set of images.
- Save and load the camera calibration data.
- Undistort images or points using the calibration data.

## Usage

Using this command to install the packages (or clone this repository):

```
pip install TH-camera-calibration
```

### Requirements

- Python 3.x
- OpenCV
- Numpy
- PyYAML (For working with YAML files)
  To set up the environment, install all the required libraries by using this command:

```
pip install -r requirements.txt
```

### How to Use

1. Import the `CameraCalibration` class from `camera_calibration.py`.

   ```python
   from camera_calibration import CameraCalibration
   ```

2. Create an instance of `CameraCalibration`.

   ```python
   camera_calibrator = CameraCalibration()
   ```

3. Calculate calibration data using a set of images with known chessboard pattern.

   ```python
   camera_calibrator.calculate_calibration_data(
       run=True,
       chessboardSize=(9, 6),
       size_of_chessboard_squares_mm=25,
       framesize=(1280, 720),
       calibrationDir=None, #path of calibration dir
       savepath="",
       saveformat="pkl",
       show_process_img=True,
       show_calibration_data=True,
   )
   ```
4. Read calibration data. (run when have calib data already and do not want to calculate calib data from scratch)
      ```python
   calibrator.read_calibration_data(r"calibration.pkl", "pkl", True)
   ```
6. Undistort an image. (If not calculate calib data from scratch, require to read calib data first)

   ```python
   undistorted_image = camera_calibrator.undistortion_img(
       img,
       method="default",
       img_size=(1280, 720),
       verbose=False
   )
   ```

   or

   ```python
   undistorted_image = camera_calibrator.undistortion_img(
       img,
       method="Remapping",
       img_size=(1280, 720),
       verbose=False
   )
   ```

## Results


## 💚🖤 Join me on Social Media 🖤💚

- My Gmail: quangtung.work73@gmail.com
- My Github: [Tung Nguyen - Handsome - Github](https://github.com/nguyenquangtung)
- My Youtube channel: [Tung Nguyen - Handsome - Youtube](https://www.youtube.com/@tungquangnguyen731)
- My Linkedin: [Tung Nguyen - Handsome - Linkedin](https://www.linkedin.com/in/tungnguyen73/)
