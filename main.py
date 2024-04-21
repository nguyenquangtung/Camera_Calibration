from camera_calibration import CameraCalibration
import cv2 as cv

img = cv.imread(r"image\results\frame_36.jpg")

calibrator = CameraCalibration()
calibrator.calculate_calibration_data(
    run=False,
    chessboardSize=(9, 6),
    size_of_chessboard_squares_mm=25,
    framesize=(1280, 720),
    calibrationDir=r"image\calibration_dir",
    savepath="",
    saveformat="pkl",
    show_process_img=False,
    show_calibration_data=False,
)
calibrator.read_calibration_data(r"calibration.pkl", "pkl", True)
# distotion_img = calibrator.remove_distortion(img, verbose=True)
points = [(1000, 250)]
new_point = calibrator.undistortion_point(points)
print(new_point)
# cv.imwrite(r"image\results\dist.jpg", distotion_img)
