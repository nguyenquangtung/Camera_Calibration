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
    outputDir="",
    saveformat="pkl",
    show_img=True,
)
calibrator.read_calibration_data(r"calibration.pkl", "pkl", True)
distotion_img = calibrator.remove_distortion(img, verbose=True)
cv.imwrite(r"image\results\dist.jpg", distotion_img)
