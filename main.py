from camera_calibration import CameraCalibration
import cv2 as cv

img = cv.imread(r"image\results\frame_36.jpg")

calibrator = CameraCalibration()
calibrator.calculate_calibration_data(
    run=True,
    chessboardSize=(9, 6),
    size_of_chessboard_squares_mm=25,
    framesize=(1280, 720),
    calibrationDir=r"image\calibration_dir",
    savepath="",
    saveformat="npz",
    show_process_img=False,
    show_calibration_data=True,
)
calibrator.read_calibration_data(r"calibration.npz", "npz", True)


################ Test undistortion img ###########################
# distortion_img = calibrator.remove_distortion(img, verbose=True)
# cv.imwrite(r"image\results\dist.jpg", distotion_img)

################ Test undistortion points ########################
# points = [(800, 200), (1200, 500)]
# new_point = calibrator.undistortion_points(points)
# print(new_point)
