import numpy as np
import cv2 as cv
import glob
import pickle
import os


class calibrate:
    def calculate_calibration_data(
        self,
        run=1,
        chessboardSize=(9, 6),
        size_of_chessboard_squares_mm=25,
        framesize=(1280, 720),
        calibrationDir=None,
        savepath=None,
        saveformat="pkl",
    ):
        if run:
            # FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS
            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
            objp[:, :2] = np.mgrid[
                0 : chessboardSize[0], 0 : chessboardSize[1]
            ].T.reshape(-1, 2)
            objp = objp * size_of_chessboard_squares_mm

            # Arrays to store object points and image points from all the images.
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.
            images = glob.glob(calibrationDir)
            for image in images:

                img = cv.imread(image)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Find the chess board corners
                cornersFound, cornersOrg = cv.findChessboardCorners(
                    gray, chessboardSize, None
                )

                # If found, add object points, image points (after refining them)
                if cornersFound == True:

                    objpoints.append(objp)
                    cornersRefined = cv.cornerSubPix(
                        gray, cornersOrg, (11, 11), (-1, -1), criteria
                    )
                    imgpoints.append(cornersRefined)

                    # Draw and display the corners
                    cv.drawChessboardCorners(
                        img, chessboardSize, cornersRefined, cornersFound
                    )
                    cv.imshow("img", img)
                    cv.waitKey(1000)
                cv.destroyAllWindows()
            # CALIBRATION
            repError, cameraMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
                objpoints, imgpoints, framesize, None, None
            )
            print("cameraMatrix: ", cameraMatrix)
            print("\Distortion Coefficent: ", distCoeff)
            self.Save_Calibration_Data(savepath, saveformat, cameraMatrix, distCoeff)
            self.Calculate_Reprojection_Error(
                objpoints, imgpoints, cameraMatrix, distCoeff, rvecs, tvecs
            )

    def Save_Calibration_Data(self, savepath, saveformat, cameraMatrix, distCoeff):
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        if saveformat == "pkl":
            with open(os.path.join(savepath, "calibration.pkl"), "wb") as f:
                pickle.dump((cameraMatrix, distCoeff), f)
        elif saveformat == "yaml":
            import yaml

            data = {
                "camera_matrix": np.asarray(cameraMatrix).tolist(),
                "dist_coeff": np.asarray(distCoeff).tolist(),
            }
            with open(os.path.join(savepath, "calibration.pkl"), "w") as f:
                yaml.dump(data, f)
        elif saveformat == "npz":
            paramPath = os.path.join(savepath, "calibration.npz")
            np.savez(paramPath, camMatrix=cameraMatrix, distCoeff=distCoeff)

    def Calculate_Reprojection_Error(
        self, objpoints, imgpoints, cameraMatrix, distCoeff, rvecs, tvecs
    ):
        # Reprojection Error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeff
            )
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("\nTotal error: {}".format(mean_error / len(objpoints)))

    def Read_Calibration_Data(self, readpath, readformat):
        if readformat == "pkl":
            with open(readpath, "rb") as f:
                pkl_data = pickle.load(f)
                cameraMatrix, distCoeff = pkl_data
        elif readformat == "yaml":
            import yaml

            with open(readpath, "r") as f:
                yaml_data = yaml.load(f, Loader=yaml.FullLoader)
                cameraMatrix, distCoeff = (
                    yaml_data["camera_matrix"],
                    yaml_data["dist_coeff"],
                )
        elif readformat == "npz":
            npz_data = np.load(readpath)
            cameraMatrix = npz_data["camMatrix"]
            distCoeff = npz_data["distCoeff"]

        return cameraMatrix, distCoeff

    def 


img = cv.imread(r"image\data1\frame_30.jpg")
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix, dist, (w, h), 0, (w, h)
)


# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# crop the image
x1, y1, w1, h1 = roi
dst = dst[y1 : y1 + h1, x1 : x1 + w1]

cv.imwrite(r"image\results\caliResult1.jpg", dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(
    cameraMatrix, dist, None, newCameraMatrix, (w, h), 5
)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x2, y2, w2, h2 = roi
dst = dst[y2 : y2 + h2, x2 : x2 + w2]
cv.imwrite(r"image\results\caliResult2.jpg", dst)


if __name__ == "__main__":
    chessboardSize = (9, 6)
    framesize = (1280, 720)
    calibrationDir = r"image\data1\*.jpg"
