import cv2
import utilz
import imgprocessing as imp
import numpy as np
import os

print(f'openCV package version: {cv2.__version__}')
data_path = utilz.path_processing('E:\git\GutMotility\data\Control_7 dpf')
file_list = os.listdir(data_path)

for video_path in file_list:
    src_path = f'{data_path}/{video_path}'
    save_path = utilz.path_processing


    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(src_path))
    if not capture.isOpened():
        print(f'Unable to open: {src_path}')
        exit(0)
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)


    ret, frame = capture.read()
    p_x, p_y, w, h = cv2.selectROI(frame)
    cv2.destroyAllWindows()

# video writer cfg
# fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
# writer = cv2.VideoWriter(f'{algorithm}_output.avi', fourcc, fps, (1920,800), isColor=False)
# origin = cv2.VideoWriter(f'_src.avi', fourcc, fps, (1920,800), isColor=False)

    # background substraction algorithm
    algorithm = 'knn'
    model = imp.BgSub_model(algorithm) 

    total_frame_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y_stmap = np.empty((h, 0), np.float64)
    x_stmap = np.empty((0, w), np.float64)

    # start = utilz.get_time()

    while True:
        ret, frame = capture.read() # (1080, 1920, 3)
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame[p_y:p_y+h, p_x:p_x+w]
        # frame = frame * mask2[p_y:p_y+h, p_x:p_x+w]

        # utilz.showInMovedWindow('frame', frame, 1000, 600)
        # origin.write(frame)

        # model-based background substraction
        fgMask = model.operate(frame)
        # cv2.putText(fgMask, utilz.calc_time_by_sec(start), (50, 100),
        #            cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255))
        # utilz.showInMovedWindow('fgMask', fgMask, 100, 100)
        # writer.write(fgMask)

        

        x_stmap = imp.make_STmap(x_stmap, fgMask, dim=0)
        y_stmap = imp.make_STmap(y_stmap, fgMask, dim=1)
        
        # keyboard = cv2.waitKey(delay)
        # if keyboard == 'q' or keyboard == 27:
        #     break


    x_stmap = x_stmap.astype(np.uint8)
    y_stmap = y_stmap.astype(np.uint8)



    x_stmap = cv2.equalizeHist(x_stmap)
    y_stmap = cv2.equalizeHist(y_stmap)

    video_name = video_path.split('.')[0]
    cv2.imwrite(f'E:/git/GutMotility/out/img/{video_name}_BS_x_stmap.png', x_stmap)
    cv2.imwrite(f'E:/git/GutMotility/out/img/{video_name}_BS_y_stmap.png', y_stmap)
    capture.release()
    # writer.release()
    # origin.release()
    cv2.destroyAllWindows()