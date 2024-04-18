from camera_calibration import CameraCalibration
import cv2 as cv

img = cv.imread(r"image\results\frame_36.jpg")
chessboardSize = (9, 6)
size_of_chessboard_squares_mm = 25
framesize = (1280, 720)
calibrationDir = r"image\calibration_dir\*.jpg"
output_img = r"image\results\dist.jpg"

calibrator = CameraCalibration()
calibrator.calculate_calibration_data(
    0,
    chessboardSize,
    size_of_chessboard_squares_mm,
    framesize,
    calibrationDir,
    "",
    "pkl",
    True,
)
calibrator.read_calibration_data(r"calibration.pkl", "pkl")
distotion_img = calibrator.remove_distortion(img)
cv.imwrite(output_img, distotion_img)
