import cv2

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

def path_processing(path:str):

    if '\\' in path:
        path = path.replace('\\', '/')
    
    return path


if __name__ == "__main__":

    path = 'E:\git\GutMotility\data\Control_7 dpf\Control_7dpf_01.mp4'
    print(path)
    print(path_processing(path))