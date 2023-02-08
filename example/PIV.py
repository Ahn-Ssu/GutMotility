import cv2, numpy as np
import utilz
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling

print(f'openCV package version: {cv2.__version__}')


# input src video cfg
input_src = 'E:/git/GutMotility/data/Control_7 dpf/Control_7dpf_04.mp4'
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(input_src))
if not capture.isOpened():
    print(f'Unable to open: {input_src}')
    exit(0)
fps = capture.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)


# background substraction algorithm
algorithm = 'KNN'
algorithm = 'MOG2'
if algorithm == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

# video writer cfg
fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
writer = cv2.VideoWriter(f'{algorithm}_output.avi', fourcc, fps, (1920,800), isColor=False)
# origin = cv2.VideoWriter(f'_src.avi', fourcc, fps, (1920,800), isColor=False)


print(f'fps: {fps}, delay: {delay}, algo: {algorithm}')
winsize = 32 # pixels, interrogation window size in frame A
searchsize = 38  # pixels, search area size in frame B
overlap = 17 # pixels, 50% overlap
dt = 0.02 # sec, time interval between the two frames
prev = None
start = utilz.get_time()
while True:
    ret, frame = capture.read() # (1080, 1920, 3)
    if frame is None:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[160:960, :] # crop
    # origin.write(frame)

    # show the origin video    
    utilz.showInMovedWindow('frame', frame, 1000, 600)

    # model-based background substraction
    frame = cv2.GaussianBlur(frame, (0, 0), 1.5)
    fgMask = backSub.apply(frame)

    if prev is None:
        prev = frame.copy()
    else:
        xs, ys, dxs, dys = utilz.vel_field(prev, frame, 16)
        norm_drs = np.sqrt(dxs ** 2 + dys ** 2)
        f = plt.figure(figsize=(20,10))

        plt.quiver(
        xs,
        ys[::-1],
        dxs,
        -dys,
        norm_drs,
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=0.1,
        )

        f.canvas.draw()
        f_arr = np.array(f.canvas.renderer._renderer)
        plt.close()

        utilz.showInMovedWindow('PIV', f_arr, -1200, 200)
    

    
    # put some info
    # cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(fgMask, utilz.calc_time_by_sec(start), (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255))
    utilz.showInMovedWindow('fgMask', fgMask, 100, 100)

    
    
    # save the processed video
    writer.write(fgMask)
    
    

    keyboard = cv2.waitKey(delay)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()
writer.release()
# origin.release()
cv2.destroyAllWindows()