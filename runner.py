import cv2
import sys
import utility
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#file selector
# app = QApplication()
# window = QFileDialog()
# file_path, _ = window.getOpenFileName()
# file_path = utility.path_processing(file_path)
# print(f'got file path = {file_path}')


# cap = cv2.VideoCapture(file_path)
cap = cv2.VideoCapture('E:/git/GutMotility/data/Control_7 dpf/Control_7dpf_03.mp4')
if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS) 
delay = int(1000/fps)
print(f'fps = {fps}, delay = {delay}')

# extracting the first frame 
ret, back = cap.read()
if not ret:
    print('Background image registration failed!')
    sys.exit()

cx, cy, cw, ch = cv2.selectROI(back)

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(back)
ROI  = cv2.rectangle(mask, (cx, cy, cw, ch), 1, -1, cv2.LINE_AA)

back = cv2.GaussianBlur(back, (0, 0), 1.0)
back = back * ROI

prev = back
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = gray * ROI
    

    diff = cv2.absdiff(gray, back)
    _, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    crop = binary[cy:cy+ch, cx:cx+cw] # shape = (y, x)

    y_sum = np.sum(diff, axis=0)
    x_sum = np.sum(diff, axis=1)
    
    for i in range(1, cnt):
        x, y, w, h, s = stats[i]
        
        if s < 100:
            continue
            
        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    
    
    
    utility.showInMovedWindow('frame', frame, 0, 0)
    utility.showInMovedWindow('mask', crop, frame.shape[1], 0)
    utility.showInMovedWindow('diff', diff+50, 0, 0)

    

    if cv2.waitKey(delay) == 27:
        break

# fig, ax = plt.subplots()


# y = []
# line, = plt.plot([])

# def update(frame):
#     y.append(frame)
#     line.set_date(y)
#     return line,

# ani = animation.FuncAnimation(fig, update, frames=x_sum)
# plt.show()


cap.release()
cv2.destroyAllWindows()
