# Camera Calibration

Created on Sunday, April 21, 2024  
Author: Tung Nguyen - Handsome  
Reference: Camera Calibration by OpenCV ([OpenCV Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html))

## Overview

This Python script provides a class, `CameraCalibration`, to perform camera calibration based on a set of image points and their corresponding object points.

## Description

This script includes functionalities to:

- Calculate camera calibration data based on a set of images.
- Save and load the camera calibration data.
- Undistort images or points using the calibration data.

## Usage

Using this command for install package (or clone this repository):

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
       calibrationDir=None,
       savepath=None,
       saveformat="pkl",
       show_process_img=True,
       show_calibration_data=True,
   )
   ```

4. Undistort an image.

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

5. Undistort a set of image points.

   ```python
   undistorted_points = camera_calibrator.undistortion_points(points, verbose=False)
   ```

## Results

## 💚🖤 Join me on Social Media 🖤💚

- My email: quangtung.work73@gmail.com
- My Github: [Tung Nguyen - Handsome - Github](https://github.com/nguyenquangtung)
- My Youtube channel: [Tung Nguyen - Handsome - Youtube](https://www.youtube.com/@tungquangnguyen731)
- My linkedin: [Tung Nguyen - Handsome - Linkedin](https://www.linkedin.com/in/tungnguyen73/)
