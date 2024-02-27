import json
import cv2
import sys
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt

# reconstruct rotation matrix from AMAZON Rekognition json file
def rotMatrix(roll,pitch,yaw):
	w = roll
	v = yaw
	u = pitch

	R = np.zeros((3,3),dtype='float')
	R[0,0] = np.cos(v)*np.cos(w)
	R[1,0] = np.cos(v)*np.sin(w)
	R[2,0] = -np.sin(v)
	R[0,1] = np.sin(u)*np.sin(v)*np.cos(w) - np.cos(u)*np.sin(w)
	R[1,1] = np.cos(u)*np.cos(w) + np.sin(u)*np.sin(v)*np.sin(w)
	R[2,1] = np.sin(u)*np.cos(v)
	R[0,2] = np.sin(u)*np.sin(w) + np.cos(u)*np.sin(v)*np.cos(w)
	R[1,2] = np.cos(u)*np.sin(v)*np.sin(w) - np.sin(u)*np.cos(w)
	R[2,2] = np.cos(u)*np.cos(v)

	return R

# reconstruct rotation matrix from AMAZON Rekognition json file
def rot_y(theta):
	R = np.zeros((3,3),dtype='float')
	R[0,0] = np.cos(theta)
	R[1,0] = 0
	R[2,0] = -np.sin(theta)
	R[0,1] = 0
	R[1,1] = 1
	R[2,1] = 0
	R[0,2] = np.sin(theta)
	R[1,2] = 0
	R[2,2] = np.cos(theta)

	return R

# obtain 3D facial position (rotation matrix) from facial parts and camera params
def estimate3Dface(nose,eyeLeft,eyeRight,mouthLeft,mouthRight,camInt,camDist):
	imgPts = np.array([eyeLeft, eyeRight, mouthLeft, mouthRight],dtype='float')
	globalPts = np.array([(-61/2,75,0), (61/2,75,0), (-47/2,0,0), (47/2,0,0)],dtype='float')			
	(success, rot_vec, trans_vec) = cv2.solvePnP(globalPts, imgPts, camInt, camDist, flags=cv2.SOLVEPNP_ITERATIVE)
	
	return rot_vec, trans_vec
	
# obtain eye (center of eye) 3D position
def eyePosition3D(rot_vec,trans_vec):
	R = cv2.Rodrigues(rot_vec)
	Rt = np.concatenate((R[0],trans_vec),axis=1)
	
	eyePos = np.array([0,75,0,1]).T
	eyePosG = np.dot(Rt,eyePos)
	
	return eyePosG

# obtain faceframe
def findFaceFrame(jsonData, ti, timewindow):
    faces = [x for x in jsonData['Face'] if x['Timestamp'] == ti]
        
    if len(faces) > 0:
        return faces
    
    faces = [x for x in jsonData['Face'] if x['Timestamp'] >= ti - timewindow and x['Timestamp'] <= ti + timewindow]
    
    if len(faces) == 0:
        return []
    
    ts = []
    for f in faces:
        ts.append(f['Timestamp'])

    array = np.array(ts)
    idx = (np.abs(array - ti)).argmin()

    faces = [x for x in jsonData['Face'] if x['Timestamp'] == ts[idx]]

    return faces
 
def getViewAngle(cam_mat, cam_dist, width, height):
    # this is rough estimate
    return np.arctan2(cam_mat[0][2],cam_mat[0][0]), np.arctan2(cam_mat[1][2],cam_mat[1][1])
 
def get_cmap():
    cmap = plt.get_cmap("tab20")
    CMAP = []
    for i in range(200):
        CMAP.append([int(cmap(i)[2]*255),int(cmap(i)[1]*255),int(cmap(i)[0]*255)])
    return CMAP 

#
#  main start from here
#
def main():

    if len(sys.argv) < 3:
        print('usage: %s [file header] [camera_parameter_file(.pkl)]\n'%(sys.argv[0]))
        sys.exit(1)

    # output options
    opt = {}
    opt['output_text'] = False
    opt['facial_marker'] = True
    opt['file_footer'] = '-result3'

    f = open(sys.argv[1]+'.json', 'r')
    jsonData = json.load(f)
    f.close()

    # open camera parameters
    with open(sys.argv[2],'rb') as f:
        [ret, cam_mtx, cam_dist, rvecs, tvecs] = pickle.load(f)

    # capture from video
    cap = cv2.VideoCapture(sys.argv[1]+'.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    view_ang = getViewAngle(cam_mtx,cam_dist,width,height)
    print('vx: ', 2*view_ang[0] * 180 / np.pi)
    print('vy: ', 2*view_ang[1] * 180 / np.pi)

    # video output
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    vout = cv2.VideoWriter(sys.argv[1] + opt['file_footer'] +'.mp4',fourcc, fps, (int((width+height)/2),int(height/2)))

    fps = float(jsonData['VideoMetadata']['FrameRate'])

    n = 0
    nfound = 0
    facedata = {}

    #timewindow = 50          # time window (amazon rekognition does not work for every frame)
    timewindow = 300          # time window (amazon rekognition does not work for every frame)
    COLS = get_cmap()
    
    while(True):
        ret, frame = cap.read()
        if frame is None:
            break
        
        if n > 0:
            ti = int((n*1000.0)/fps)
        else:
            ti = 0
        
        faces = findFaceFrame(jsonData, ti, timewindow)
        n += 1

        # map image
        s = 0.1        
        mimage = np.zeros((height,height,3),dtype='uint8')
        wx = mimage.shape[1]    # image size of map
        wy = mimage.shape[0]
        for r in range(0, 10000, 500):
            cv2.circle(mimage, (int(wx*0.5),wy), int(r*s), (100,100,100), 1,cv2.LINE_4)  
       # draw viewing angles in map
        cv2.line(mimage,(int(wx*0.5),wy),
            (int(wx*0.5 + np.cos(0.5*np.pi - view_ang[0])*5000), int(wy - np.sin(0.5*np.pi - view_ang[0])*5000)),
            (100,100,100), 2)
        cv2.line(mimage,(int(wx*0.5),wy),
            (int(wx*0.5 - np.cos(0.5*np.pi - view_ang[0])*5000), int(wy - np.sin(0.5*np.pi - view_ang[0])*5000)),
            (100,100,100), 2)               
    
        # prepare facedata list
        facedata[str(n)] = []

        for nf,face in enumerate(faces):
            if nf >= len(COLS):
                continue
                
            nfound += 1
            face = face['Face']
            
            # draw boundingbox
            w = frame.shape[1]
            h = frame.shape[0]
            lt = (int(face['BoundingBox']['Left']*w), int(face['BoundingBox']['Top']*h))
            rb = (int(lt[0]+face['BoundingBox']['Width']*w),int(lt[1]+face['BoundingBox']['Height']*h))
            cv2.rectangle(frame, lt, rb, COLS[nf])
            
            # draw facial parts
            for lms in face['Landmarks']:
                le = (int(lms['X']*w),int(lms['Y']*h))
                
                if opt['facial_marker'] == True:
                    cv2.circle(frame, le, 5, (0,255,255), thickness=3, lineType=cv2.LINE_AA)
                
                if lms['Type'] == 'nose':
                    nose = np.array(le)
                if lms['Type'] == 'eyeRight':
                    eyeRight = np.array(le)
                if lms['Type'] == 'eyeLeft':
                    eyeLeft = np.array(le)
                if lms['Type'] == 'mouthRight':
                    mouthRight = np.array(le)
                if lms['Type'] == 'mouthLeft':
                    mouthLeft = np.array(le)					
            
            # rotation parameters
            yaw  = face['Pose']['Yaw']
            roll = face['Pose']['Roll']
            pitch = face['Pose']['Pitch']
            
            fcenter = (eyeRight + eyeLeft + mouthRight + mouthLeft)/4
            fcenter = (int(fcenter[0]),int(fcenter[1]))
            
            yaw = yaw*np.pi / 180.0
            roll = roll*np.pi / 180.0
            pitch = pitch*np.pi / 180.0     
            R = rotMatrix(roll,pitch,yaw)
            
            # draw facial coordinate
            X = np.array([200,0,0])
            Y = np.array([0,200,0])
            Z = np.array([0,0,200])
            X = np.dot(R,X.T)
            Y = np.dot(R,Y.T)
            Z = np.dot(R,Z.T)
            
            cv2.line(frame,fcenter,(fcenter[0]+int(Z[0]),fcenter[1]+int(Z[1])),COLS[nf],3)
                    
            # estimate 3D position of the face
            face_rot, face_trans = estimate3Dface( nose, eyeLeft, eyeRight, mouthLeft, mouthRight, cam_mtx,cam_dist)
            
            # draw facial markers
            x = int(wx/2 + s*face_trans[0])
            z = int(wy - s*face_trans[2])
            
            #
            # adjust facial direction considering camera rotation
            #
            theta = np.pi/2 - np.arctan2(face_trans[2],face_trans[0])
            R = rot_y(-theta)
            # print(R)
            ZZ = np.dot(R,Z.T)
            #vx = int(Z[0])
            #vz = int(Z[2])
            try:
                cv2.circle(mimage, (x,z), 15, COLS[nf], 3, cv2.LINE_4)
            except:
                print('overflow error')
                
            # draw non-adjusted facial direction (assuming orthogonal)
            #cv2.line(mimage, (x,z), (x+vx, z+vz), (255,255,255), 2)
            vx = int(ZZ[0])
            vz = int(ZZ[2])
            # draw adjusted facial direction
            try:            
                cv2.line(mimage, (x,z), (x+vx, z+vz), (255,255,255), 2)
            except:
                print('overflow error')
                
            if opt['output_text'] == True:
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, \
                    'Frame %d: Facepos: %4.2f  %4.2f %4.2f'%(n,face_trans[0],face_trans[1],face_trans[2]), \
                    (12,30*(nf+1)), font, 2, COLS[nf], 2)
                    
            f = {}
            f['success'] = 1
            f['face_Tx'] = face_trans[0]
            f['face_Ty'] = face_trans[1]
            f['face_Tz'] = face_trans[2]
            f['yaw'] = yaw
            f['roll'] = roll
            f['pitch'] = pitch
            f['ZZ_X'] = ZZ[0]
            f['ZZ_Y'] = ZZ[1]
            f['ZZ_Z'] = ZZ[2]
            facedata[str(n)].append(f)      

        outframe = cv2.resize(np.hstack([frame,mimage]), dsize=None, fx=0.5, fy=0.5)
 
        vout.write(outframe)
        #cv2.imshow("monitor", np.hstack([frame,mimage]))
        cv2.imshow("monitor", outframe)
        
        k = cv2.waitKey(5)
        if k == 32:         # space key
            print('pause')
            cv2.waitKey(0)
        elif k == 27:       # esc key
            break

    # output facial output files
    nfound = 0
    with open(sys.argv[1] + '.tsv', 'w') as f:
        f.write('Frame\tFaceID\tsuccess\tface_Tx\tface_Ty\tface_Tz\tyaw\troll\tpitch\tz_adj_x\tz_adj_y\tz_adj_z\n')
        
        for k in facedata.keys():
            for nf,fd in enumerate(facedata[k]):
                f.write('%s\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'%(k, nf, fd['success'], \
                                fd['face_Tx'],fd['face_Ty'],fd['face_Tz'], \
                                fd['yaw'],fd['roll'],fd['pitch'],fd['ZZ_X'],fd['ZZ_Y'],fd['ZZ_Z']))
            nfound += 1

    cap.release()
    vout.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: %s [file header]\n'%(sys.argv[0]))
        sys.exit(1)

    main()
