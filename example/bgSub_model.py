from __future__ import print_function
import cv2 as cv
import utility


input_src = 'E:/git/GutMotility/data/Control_7 dpf/Control_7dpf_03.mp4'
algorithm = 'MOG2' # 'KNN
if algorithm == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(input_src))
if not capture.isOpened():
    print('Unable to open: ' + input_src)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    frame = cv.GaussianBlur(frame, (0, 0), 2)
    
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    utility.showInMovedWindow('frame', frame, 800, 600)
    utility.showInMovedWindow('fgMask', fgMask, 0, -250)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break