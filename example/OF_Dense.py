import numpy as np
import cv2 
import utilz

cap = cv2.VideoCapture('E:/git/GutMotility/data/Control_7 dpf/Control_7dpf_03.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
writer = cv2.VideoWriter(f'OF_dense_output.avi', fourcc, fps, (1920,800))

ret, frame1 = cap.read()
frame1 = frame1[160:960, :, :]
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

start = utilz.get_time()
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame2 = frame2[160:960, :, :]
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    
    utilz.showInMovedWindow('frame', frame2, 800, 600)

    cv2.putText(bgr, utilz.calc_time_by_sec(start), (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255))
    utilz.showInMovedWindow('Dense', bgr, 100, 100)
    writer.write(bgr)
    



    k = cv2.waitKey(delay) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', bgr)
    prvs = next


writer.release()
cap.release()
cv2.destroyAllWindows()