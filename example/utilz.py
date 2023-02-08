import cv2
import numpy as np
from scipy.signal import correlate
import openpiv



def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

def path_processing(path:str):

    if '\\' in path:
        path = path.replace('\\', '/')
    
    return path

def get_time():
    import time
    return time.time()

def calc_time_by_sec(start):
    import datetime
    delta = (get_time() - start)
    out = str(datetime.timedelta(seconds=delta))[:-4]

    return out

def vel_field(curr_frame, next_frame, win_size):
    ys = np.arange(0, curr_frame.shape[0], win_size)
    xs = np.arange(0, curr_frame.shape[1], win_size)
    dys = np.zeros((len(ys), len(xs)))
    dxs = np.zeros((len(ys), len(xs)))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            int_win = curr_frame[y : y + win_size, x : x + win_size]
            search_win = next_frame[y : y + win_size, x : x + win_size]
            cross_corr = correlate(
                search_win - search_win.mean(), int_win - int_win.mean(), method="fft"
            )
            dys[iy, ix], dxs[iy, ix] = (
                np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
                - np.array([win_size, win_size])
                + 1
            )
    # draw velocity vectors from the center of each window
    ys = ys + win_size / 2
    xs = xs + win_size / 2
    return xs, ys, dxs, dys

if __name__ == "__main__":

    path = 'E:\git\GutMotility\data\Control_7 dpf\Control_7dpf_01.mp4'
    print(path)
    print(path_processing(path))