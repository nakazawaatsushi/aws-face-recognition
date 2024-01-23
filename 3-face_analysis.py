import json
import cv2
import sys
import numpy as np
import csv

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

# obtain 3D facial position (rotation matrix) from facial parts and camera params
def estimate3Dface(nose,eyeLeft,eyeRight,mouthLeft,mouthRight,camInt,camDist):
	imgPts = np.array([eyeLeft, eyeRight, mouthLeft, mouthRight],dtype='float')
	globalPts = np.array(\
					[(-61/2,75,0),\
					(61/2,75,0), \
					(-47/2,0,0),\
						(47/2,0,0)],dtype='float')			
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
    faces = [x for x in jsonData['Face'] if x['Timestamp'] >= ti - timewindow and x['Timestamp'] <= ti + timewindow]
    
    if len(faces) == 0:
        return []
        
    print(ti, faces[0]['Timestamp'])
    
    
    return faces
 
#
#  main start from here
#
if len(sys.argv) < 3:
    print('usage: %s [file header] [camera matrix header]\n'%(sys.argv[0]))
    sys.exit(1)

f = open(sys.argv[1]+'.json', 'r')
jsonData = json.load(f)
f.close()

camInt  = np.loadtxt("%s-int.csv"%(sys.argv[2]),delimiter=",")
camDist = np.loadtxt("%s-dist.csv"%(sys.argv[2]),delimiter=",")

# capture from video
cap = cv2.VideoCapture(sys.argv[1]+'.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# video output
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
vout = cv2.VideoWriter(sys.argv[1]+'-result.mp4',fourcc, fps, (width,height))

fps = float(jsonData['VideoMetadata']['FrameRate'])

n = 0
nfound = 0
facedata = {}

timewindow = 50          # time window (amazon rekognition does not work for every frame)

COLS = [(0,255,0),(0,0,255), (255,0,0), (255,255,0), (0,255,255),(255,255,255),(255,0,255),
        (0,255,0),(0,0,255), (255,0,0), (255,255,0), (0,255,255),(255,255,255),(255,0,255),
        (0,255,0),(0,0,255), (255,0,0), (255,255,0), (0,255,255),(255,255,255),(255,0,255)]

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
        X = np.array([100,0,0])
        Y = np.array([0,100,0])
        Z = np.array([0,0,100])
        X = np.dot(R,X.T)
        Y = np.dot(R,Y.T)
        Z = np.dot(R,Z.T)
        
        cv2.line(frame,fcenter,(fcenter[0]+int(X[0]),fcenter[1]+int(X[1])),(255,0,0),3)
        cv2.line(frame,fcenter,(fcenter[0]+int(Y[0]),fcenter[1]+int(Y[1])),(0,255,0),3)
        cv2.line(frame,fcenter,(fcenter[0]+int(Z[0]),fcenter[1]+int(Z[1])),(0,0,255),3)
                
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, \
            'Frame %d: Roll,Pitch,Yaw = %f,%f,%f'%(n,roll*180/np.pi,pitch*180/np.pi,yaw*180/np.pi), \
            (12,20), font, 1, (255,255,255), 2)
    
    vout.write(frame)
    cv2.imshow("monitor",frame)
    cv2.waitKey(0)

with open(sys.argv[1] + '-all.csv', 'w') as f:
    f.write('Frame,success,pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz\n')
    
    for k in facedata.keys():
        if facedata[k]['success'] == 0:
            f.write('%s,%d\n'%(k,facedata[k]['success']))
        else:
            f.write('%s,%d,%f,%f,%f,%f,%f,%f\n'%(k,facedata[k]['success'], \
                            facedata[k]['pose_Tx'],facedata[k]['pose_Ty'],facedata[k]['pose_Tz'], \
                            facedata[k]['pose_Rx'],facedata[k]['pose_Ry'],facedata[k]['pose_Rz']))
    
    f.write('#Face is found %d frames / %d frames.'%(nfound,n))

# for web visualization
NSMPL = 200
STEP = (len(facedata)+1)/NSMPL

with open(sys.argv[1] + '-face_dist.csv', 'w') as f:
    for k in facedata.keys():
        if int(k)%STEP >= 0.0 and int(k)%STEP < 1.0:
            if facedata[k]['success'] == 0:
                f.write('0.0\n')
            else:
                f.write('%f\n'%(facedata[k]['pose_Tz']))

with open(sys.argv[1] + '-face_rx.csv', 'w') as f:
    for k in facedata.keys():
        if int(k)%STEP >= 0.0 and int(k)%STEP < 1.0:
            if facedata[k]['success'] == 0:
                f.write('0.0\n')
            else:
                f.write('%f\n'%(facedata[k]['pose_Rx']))

with open(sys.argv[1] + '-face_ry.csv', 'w') as f:
    for k in facedata.keys():
        if int(k)%STEP >= 0.0 and int(k)%STEP < 1.0:
            if facedata[k]['success'] == 0:
                f.write('0.0\n')
            else:
                f.write('%f\n'%(facedata[k]['pose_Ry']))

with open(sys.argv[1] + '-face_rz.csv', 'w') as f:
    for k in facedata.keys():
        if int(k)%STEP >= 0.0 and int(k)%STEP < 1.0:
            if facedata[k]['success'] == 0:
                f.write('0.0\n')
            else:
                f.write('%f\n'%(facedata[k]['pose_Rz']))
    
print('Face is found %d frames / %d frames.'%(nfound,n))    

cap.release()
vout.release()
cv2.destroyAllWindows()
