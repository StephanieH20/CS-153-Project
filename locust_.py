from cmath import nan
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi
from scipy.io import loadmat
import random
import os

seq = -1
while seq < 0 or seq == 8 or seq > 9:
    seq = int(input("What sequence? "))
    if seq < 0 or seq == 8 or seq > 9:
        print("Invalid sequence")

data = loadmat('sequence_' + str(seq) + '/frame_1.mat')

frms = 250
if seq != 7 and seq != 9:
    frms = 1500

vidArr = [
'video_clips/preprocessed_133_1min_225.avi', 
'video_clips/preprocessed_133_1min_650.avi', 
'video_clips/preprocessed_133_1min_750.avi', 
'video_clips/preprocessed_133_1min_850.avi', 
'video_clips/preprocessed_133_1min_950.avi', 
'video_clips/preprocessed_133_1min_1115.avi', 
'video_clips/preprocessed_133_10sec_710.avi',
nan,
'video_clips/preprocessed_133_10sec_1910.avi', 
]

cap = cv2.VideoCapture(vidArr[seq - 1])

#data = loadmat('sequence_9/frame_1.mat')

featarray = data['featarray']

angles = [-1] * 500

width = int(cap.get(3))
height = int(cap.get(4))

scales = [
(width/100, height/56.25), 
(width/100, height/56.25), 
(width/100, height/56.25), 
(width/100, height/56.25), 
(width/100, height/56.25), 
(width/100, height/56.25), 
(width/100, height/56.25), 
nan,
(width/100, height/56.25),
]

translations = [
(0, 0),
(0, 0), 
(0, 0), 
(0, 0), 
(0, 0), 
(0, 0), 
(0, 0), 
nan,
(0, 0)
]

pts = []

_, frm = cap.read()
old = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
first = old.copy()

thresh = cv2.threshold(first, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
connComp = cv2.connectedComponentsWithStats(thresh, 1, cv2.CV_32S)
(num, labels, stats, centroids) = connComp

for i in range (1, num):
    pts.append((int(centroids[i][0]),int(centroids[i][1])))

old_pts = np.array(pts, dtype = np.float32)

index = 120

rands = []

t_d = 3

for i in range(0, 500):
    rands.append((random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)))

accuracies = []
o_accuracies = []

n_accuracy = 0
n_o_accuracy = 0
t_locs = 0

def dist(p1, p2, p3, p4):
    return sqrt(((p1-p3)*(p1-p3))+((p2-p4)*(p2-p4)))

def getAngle(img, center, angle, scale, n1, n2):
    hypotenuse = 5
    length = dist(center[0], center[1], n1, n2)

    p1 = int(center[0] + length * cos(angle))
    p2 = int(center[1] + length * sin(angle))

    p1_2 = int(center[0] - length * cos(angle))
    p2_2 = int(center[1] - length * sin(angle))

    if dist(p1, p2, n1, n2) <= dist(p1_2, p2_2, n1, n2):
        cv2.line(img, (center), (int(center[0] + scale * hypotenuse * cos(angle)), int(center[1] + scale * hypotenuse * sin(angle))), (0, 255, 0), 2, cv2.LINE_AA)
        return angle
    else:
        cv2.line(img, (center), (int(center[0] - scale * hypotenuse * cos(angle)), int(center[1] - scale * hypotenuse * sin(angle))), (0, 255, 0), 2, cv2.LINE_AA)
    return angle + pi

def comp_angles(a1, a2):
    er = (abs(a1 - a2) * 100) % (200 * pi)
    acc = 1 - (min(er, (200 * pi) - er)/(100 * pi))
    return acc

def comp_orientations(a1, a2):
    return max(comp_angles(a1, a2), comp_angles(a1 + pi, a2))


def getOrientation(pts, img, n_x, n_y):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    calc_angle = getAngle(img, cntr, angle, 5, n_x, n_y)
    
    return (cntr, angle, calc_angle)

ct = 1
curr_pos = [[] for i in range(2000)]
test_t = 0
min_x = 10000
min_y = 10000
max_x = 0
max_y = 0

while ct <= frms:
    data_f = loadmat('sequence_' + str(seq) + '/frame_' + str(ct) + '.mat')
    #data_f = loadmat('sequence_9/frame_' + str(ct) + '.mat')
    featarray_f = data_f['featarray']
    ct = ct + 1
    for i in range(0, 2000):
        curr_pos[i].append((nan, nan))
    for i in range(0, len(featarray_f)):
        x_p = featarray_f[i][0]*scales[seq-1][0]
        y_p = featarray_f[i][1]*scales[seq-1][1]
        I_D = int(featarray_f[i][9])
        if len(curr_pos[I_D]) > 0:
            curr_pos[I_D].pop()
        curr_pos[I_D].append((x_p, y_p))

direcName = "seq_" + str(seq) + "_results"

if not os.path.isdir(direcName):
    os.mkdir(direcName)

count = 1

while count < frms:
    x_scale = scales[seq-1][0]
    y_scale = scales[seq-1][1]
    x_translation = translations[seq-1][0]
    y_translation = translations[seq-1][1]
    accuracy = 0
    o_accuracy = 0
    locs = 0

    compName = os.path.join(direcName, "frame_" + str(count) + ".txt") 
    fil = open(compName, "w")

    if count == 1:
        frame = frm

    else:
        ret, frame = cap.read()  
        
    curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    data = loadmat('sequence_' + str(seq) + '/frame_' + str(count) + '.mat')
    #data = loadmat('sequence_9/frame_' + str(count) + '.mat')
    
    featarray = data['featarray']

    thresh = cv2.threshold(curr, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    connComp = cv2.connectedComponentsWithStats(thresh, 1, cv2.CV_32S)
    (num, labels, stats, centroids) = connComp

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hulls = []
    boxes = []
    box_es = []

    for i in range(0, len(contours)):
        c_length = cv2.arcLength(contours[i], True)
        if c_length > 0:
            hull = cv2.convexHull(contours[i])
            hulls.append(hull)
            rect = cv2.minAreaRect(contours[i])
            box_ = cv2.boxPoints(rect)
            box_ = np.int0(box_)
            box_es.append(box_)
            x,y,w,h = cv2.boundingRect(contours[i])
            box = [[x-t_d, y-t_d], [x+w+t_d, y-t_d], [x+w+t_d, y+h+t_d], [x-t_d, y+h+t_d]]
            box = np.int0(box)
            boxes.append(box)

    for j in range(0, len(featarray)):
        
        #cv2.circle(frame, (int(featarray[j][0]-x_translation), int(featarray[j][1]-y_translation)), 5, (0, 255, 0), 2)
        
        px, py = int((featarray[j][0]-x_translation)*x_scale), int((featarray[j][1]-y_translation)*y_scale)
        #cv2.circle(frame, (px, py), 5, (0, 255, 0), 2)
        max_x = max(featarray[j][0], max_x)
        max_y = max(featarray[j][1], max_y)
        min_x = min(featarray[j][0], min_x)
        min_y = min(featarray[j][1], min_y)
        #print(max_x, max_y)
        #print(min_x, min_y)

        #print(max_x-min_x, max_y-min_y)
        
        for k in range(0, len(hulls)):
            if cv2.pointPolygonTest(boxes[k], (int(featarray[j][0]*x_scale), int(featarray[j][1]*y_scale)), True) >= 0:
                eyeDee = int(featarray[j][9])
                curr_x, curr_y = int(featarray[j][0]*x_scale), int(featarray[j][1]*y_scale)
                next_x, next_y = curr_pos[eyeDee][count]

                if np.isnan(next_x):
                    next_x = curr_x
                    next_y = curr_y

                actual_angle = featarray[j][4]

                (center, angle, calc_angle) = getOrientation(hulls[k], frame, next_x, next_y)
                cv2.circle(frame, center, 10, (0, 255, 0), 2)

                #print(str(eyeDee) + ": " + "(" + str(center[0]/x_scale) + ", " + str(center[1]/y_scale) + ")  " + str(calc_angle))
                
                fil.write(str(eyeDee) + ": " + "(" + str(center[0]/x_scale) + ", " + str(center[1]/y_scale) + ")  " + str(calc_angle) + "\n")


                if not np.isnan(actual_angle):
                    comp_a = comp_angles(calc_angle, actual_angle)
                    comp_o = comp_orientations(calc_angle, actual_angle)
                    if featarray[j][8] == 0:
                        #cv2.line(frame, (center), (int(center[0] + 25 * cos(actual_angle)), (int(center[1] + 25 * sin(actual_angle)))), (0, 100, 155), 2, cv2.LINE_AA)
                        n_accuracy = n_accuracy + ((0.5) * comp_a)
                        n_o_accuracy = n_o_accuracy + ((0.5) * comp_o)
                        accuracy = accuracy + ((0.5) * comp_a)
                        o_accuracy = o_accuracy + ((0.5) * comp_o)
                        locs = locs + 0.5
                        t_locs = t_locs + 0.5
                    elif featarray[j][8] == 1:
                        #cv2.line(frame, (center), (int(center[0] + 25 * cos(actual_angle)), (int(center[1] + 25 * sin(actual_angle)))), (0, 0, 255), 2, cv2.LINE_AA)
                        n_accuracy = n_accuracy + comp_a
                        n_o_accuracy = n_o_accuracy + comp_o
                        accuracy = accuracy + comp_a
                        o_accuracy = o_accuracy + comp_o
                        locs = locs + 1
                        t_locs = t_locs + 1
                    elif featarray[j][8] == 2:
                        #cv2.line(frame, (center), (int(center[0] + 25 * cos(actual_angle)), (int(center[1] + 25 * sin(actual_angle)))), (0, 50, 205), 2, cv2.LINE_AA)
                        n_accuracy = n_accuracy + ((0.75) * comp_a)
                        n_o_accuracy = n_o_accuracy + ((0.75) * comp_o)
                        accuracy = accuracy + ((0.75) * comp_a)
                        o_accuracy = o_accuracy + ((0.75) * comp_o)
                        locs = locs + 0.75
                        t_locs = t_locs + 0.75
                    
                continue

    if locs > 0:
        accuracy = accuracy/locs
        accuracies.append(accuracy)

        o_accuracy = o_accuracy/locs
        o_accuracies.append(o_accuracy)

    #print(str(count) + "------------------------")
    count = count + 1
    

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

print("Orientation accuracy: " + str(n_accuracy/t_locs))
print("Axis accuracy: " + str(n_o_accuracy/t_locs))

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

x = range(0,len(accuracies))
plt.plot(x, accuracies)
plt.ylim([0, 1])
plt.show()
