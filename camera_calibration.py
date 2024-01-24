import numpy as np
import cv2 as cv
import glob
import sys
import os
import pickle

def main():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # number of grid points
    CNR_Y = 10
    CNR_X = 6
    GRID_SIZE_IN_MM = 20.0
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CNR_Y*CNR_X,3), np.float32)
    objp[:,:2] = np.mgrid[0:CNR_X,0:CNR_Y].T.reshape(-1,2) * GRID_SIZE_IN_MM
    print(objp)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(os.path.join(sys.argv[1],'*.png'))

    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (CNR_X,CNR_Y), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (CNR_X,CNR_Y), corners2, ret)
            cv.imshow('img', img)
            
        if cv.waitKey(50) == 27:
            break
            
    print('now performing calibration...')
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret, mtx, dist, rvecs, tvecs)
    
    with open(os.path.join(sys.argv[1],'camera_parameter.pkl'), 'wb') as f:
        pickle.dump([ret, mtx, dist, rvecs, tvecs], f)
    
    print('file saved as ', os.path.join(sys.argv[1],'camera_parameter.pkl'))
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: %s [file header] [CNR_X(optional)] [CNR_Y(optional)]\n'%(sys.argv[0]))
        sys.exit(1)

    main()