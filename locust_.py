from cmath import nan
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi
from scipy.io import loadmat
import random

cap = cv2.VideoCapture('video_clips/preprocessed_133_10sec_710.avi')
#cap = cv2.VideoCapture('video_clips/preprocessed_133_10sec_1910.avi')

data = loadmat('sequence_7/frame_1.mat')
#data = loadmat('sequence_9/frame_1.mat')

featarray = data['featarray']

angles = [-1] * 500

width = int(cap.get(3))
height = int(cap.get(4))

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
    # return (cos(a1 - a2)/2) + 0.5

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
curr_pos = [[] for i in range(500)]
test_t = 0
while(ct < 250):
    data_f = loadmat('sequence_7/frame_' + str(ct) + '.mat')
    #data_f = loadmat('sequence_9/frame_' + str(ct) + '.mat')
    featarray_f = data_f['featarray']
    ct = ct + 1
    for i in range(0, 500):
        curr_pos[i].append((nan, nan))
    for i in range(0, len(featarray_f)):
        x_p = featarray_f[i][0]*width/100
        y_p = featarray_f[i][1]*height/56.25
        I_D = int(featarray_f[i][9])
        curr_pos[I_D].pop()
        curr_pos[I_D].append((x_p, y_p))


count = 1

while count < 248:
    accuracy = 0
    o_accuracy = 0
    locs = 0

    ret, frame = cap.read()  
    curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    count = count + 1
    
    data = loadmat('sequence_7/frame_' + str(count) + '.mat')
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
        if c_length > 20:
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
        px, py = int(featarray[j][0]*width/100), int(featarray[j][1]*height/56.25)
        for k in range(0, len(hulls)):
            if cv2.pointPolygonTest(boxes[k], (int(featarray[j][0]*width/100), int(featarray[j][1]*height/56.25)), True) >= 0:
                eyeDee = int(featarray[j][9])
                curr_x, curr_y = int(featarray[j][0]*width/100), int(featarray[j][1]*height/56.25)
                next_x, next_y = curr_pos[eyeDee][count]

                if np.isnan(next_x):
                    next_x = curr_x
                    next_y = curr_y

                actual_angle = featarray[j][4]

                (center, angle, calc_angle) = getOrientation(hulls[k], frame, next_x, next_y)
                cv2.circle(frame, center, 10, (0, 255, 0), 2)
                
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

    accuracy = accuracy/locs
    accuracies.append(accuracy)

    o_accuracy = o_accuracy/locs
    o_accuracies.append(o_accuracy)

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