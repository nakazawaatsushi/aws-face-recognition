import json
import cv2
import sys
import numpy as np
import csv
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
    
    """
    face1 = [x for x in jsonData['Face'] if x['Timestamp'] >= ti - timewindow and x['Timestamp'] <= ti]
    face2 = [x for x in jsonData['Face'] if x['Timestamp'] <= ti + timewindow and x['Timestamp'] > ti]
    
    if len(face1) > 0 and len(face2) > 0:
    
        # perform interpolation
        #print(face1)
        t1 = int(face1[0]['Timestamp'])
        t2 = int(face2[0]['Timestamp'])
        face = []
        face.append({})
        r = (ti - t1)/(t2 - t1)
        
        face[0]['Timestamp'] = ti
        face[0]['Face'] = {}
        face[0]['Face']['BoundingBox'] = {}
        face[0]['Face']['Landmarks'] = []
        face[0]['Face']['Pose'] = {}
        
        for k in face1[0]['Face']['BoundingBox'].keys():
            face[0]['Face']['BoundingBox'][k] = (1-r)*face1[0]['Face']['BoundingBox'][k] + r*face2[0]['Face']['BoundingBox'][k]
            
        for i in range(len(face1[0]['Face']['Landmarks'])):
            tmp = {} 
            tmp['Type'] = face1[0]['Face']['Landmarks'][i]['Type']
            tmp['X'] = face1[0]['Face']['Landmarks'][i]['X']
            tmp['Y'] = face1[0]['Face']['Landmarks'][i]['Y']            
            face[0]['Face']['Landmarks'].append(tmp)
                       
        for k in face1[0]['Face']['Pose']:
            #print(k)
            #print(face1[0]['Face']['Pose'][k])
            face[0]['Face']['Pose'][k] = (1-r)*face1[0]['Face']['Pose'][k] + r*face2[0]['Face']['Pose'][k]
        
        #parts = ['eyeRight','eyeLeft','mouthRight','mouthLeft','nose']
        #for p in parts:
        #    face[0]['Face']['Landmarks'][p]['X'] = (1-r)*face1[0]['Face']['Landmarks'][p]['X'] + r*face2[0]['Face']['Landmarks'][p]['X']
        #    face[0]['Face']['Landmarks'][p]['Y'] = (1-r)*face1[0]['Face']['Landmarks'][p]['Y'] + r*face2[0]['Face']['Landmarks'][p]['Y']            
        #
        #parts = ['Yaw','Roll','Pitch']
        #for p in parts:
        #    face[0]['Face']['Pose'][p] = (1-r)*face1[0]['Face']['Pose'][p] + r*face2[0]['Face']['Pose'][p]   
    
    return face
    """    
def get_cmap():
    cmap = plt.get_cmap("Paired")
    CMAP = []
    for i in range(100):
        CMAP.append([int(cmap(i)[2]*255),int(cmap(i)[1]*255),int(cmap(i)[0]*255)])
    return CMAP
    
#
#  main start from here
#
if len(sys.argv) < 2:
    print('usage: %s [file header]\n'%(sys.argv[0]))
    sys.exit(1)

f = open(sys.argv[1]+'.json', 'r')
jsonData = json.load(f)
f.close()

# capture from video
cap = cv2.VideoCapture(sys.argv[1]+'.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# video output
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
vout = cv2.VideoWriter(sys.argv[1]+'-result.mp4',fourcc, fps, (int((width+height)/2),int(height/2)))

n = 0
facedata = {}

timewindow = 300          # time window (amazon rekognition does not work for every frame)

COLS = [(0,255,0),(0,0,255), (255,255,0), (0,255,255),(255,255,255),(255,0,255),(0,128,0),(0,0,128),(128,128,0), (0,128,128),(128,0,128)]
COLS = get_cmap()


while(True):
    ret, frame = cap.read()
    
    if frame is None:
        break
    
    if n > 0:
        ti = int(n*1000.0/fps)
    else:
        ti = 0
    
    # find nearest frame or interpolate (to be implemented in the future)
    faces = findFaceFrame(jsonData, ti, timewindow)
    #faces = [x for x in jsonData['Face'] if x['Timestamp'] >= ti - timewindow and x['Timestamp'] <= ti + timewindow]

    # map image
    mimage = np.zeros((height,height,3),dtype='uint8')

    facedata[str(n)] = []

    #print('length = ', len(faces))
    for nf,face in enumerate(faces):
        if nf >= len(COLS):
            continue
   
        face = face['Face']
        
        # draw boundingbox
        w = frame.shape[1]
        h = frame.shape[0]
        lt = (int(face['BoundingBox']['Left']*w), int(face['BoundingBox']['Top']*h))
        rb = (int(lt[0]+face['BoundingBox']['Width']*w),int(lt[1]+face['BoundingBox']['Height']*h))
        cv2.rectangle(frame, lt, rb, COLS[nf], thickness=2)
        
        # draw facial parts
        for lms in face['Landmarks']:
            le = (int(lms['X']*w),int(lms['Y']*h))
            # cv2.circle(frame, le, 5, (0,255,255), thickness=3, lineType=cv2.LINE_AA)
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

        #print('roll,pitch,yaw = ',roll,pitch,yaw)
        
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
        
        cv2.line(frame,fcenter,(fcenter[0]+int(Z[0]),fcenter[1]+int(Z[1])),COLS[nf],3)

        thetaDist = np.zeros(5)
        
        # theta camera parameter for image size 1920 x 990
        thetaInt = np.array([[304.3396, 0, 0],[0, 300.0791, 0], [fcenter[0], fcenter[1], 1.0000]])
        #thetaInt = np.array([[1, 0, fcenter[0]],[0, 1, fcenter[1]], [0, 0, 1]])

        # perform depth estimation
        rot,trans = estimate3Dface(nose,eyeLeft,eyeRight,mouthLeft,mouthRight,thetaInt,thetaDist)
        eyePos = eyePosition3D(rot,trans)
        #print('Eye Position = ',eyePos)
        
        # facial position in world coordinate, use only distance information for eyePos
        offset = np.pi;
        eyePos_w = [np.cos(np.pi-2*np.pi*fcenter[0]/w+offset)*eyePos[2], 
                    np.sin(np.pi-2*np.pi*fcenter[0]/w+offset)*eyePos[2]]
        yaw_w = yaw - 2*np.pi*fcenter[0]/w + offset
        
        s = 0.1
        x = int(mimage.shape[0]/2 + s*eyePos_w[0])
        y = int(mimage.shape[1]/2 - s*eyePos_w[1])
        vx = int(np.cos(yaw_w)*200)
        vy = int(np.sin(yaw_w)*200)
        cv2.circle(mimage, (x,y), 15, COLS[nf], 3, cv2.LINE_4)
        cv2.line(mimage, (x,y), (x+vx, y-vy), (128,128,128), 3)
               
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, \
            'Frame %d: Facepos: %4.2f  %4.2f  FaceDirec: %3.2f'%(n,eyePos_w[0],eyePos_w[1],180*yaw_w/np.pi), \
            (12,30*(nf+1)), font, 2, COLS[nf], 2)
        f = {}
        f['success'] = 1
        f['face_Tx'] = eyePos_w[0]
        f['face_Ty'] = eyePos_w[1]
        f['yaw'] = yaw
        f['roll'] = roll
        f['pitch'] = pitch
        f['yaw_w'] = yaw_w
        facedata[str(n)].append(f)
    
    outframe = cv2.resize(np.hstack([frame,mimage]), dsize=None, fx=0.5, fy=0.5)
    cv2.imshow('output', outframe)
    
    if cv2.waitKey(1) == 27:
        break;
        
    vout.write(outframe)
    n += 1

cap.release()
vout.release()

nfound = 0
with open(sys.argv[1] + '.tsv', 'w') as f:
    f.write('Frame\tFaceID\tsuccess\tface_Tx\tface_Ty\tyaw\troll\tpitch\tyaw_w\n')
    
    for k in facedata.keys():
        for nf,fd in enumerate(facedata[k]):
            f.write('%s\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n'%(k,nf,fd['success'], \
                            fd['face_Tx'],fd['face_Ty'],fd['yaw'], \
                            fd['roll'],fd['pitch'],fd['yaw_w']))
        nfound += 1
    
    f.write('#Face is found %d frames / %d frames.'%(nfound,n))

"""
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
"""
    
print('Face is found %d frames / %d frames.'%(nfound,n))    


cv2.destroyAllWindows()
