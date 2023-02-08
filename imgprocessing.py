import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BgSub_model():
    def __init__(self, algorithm:str) -> None:
        assert algorithm in ['knn', 'mog2'], 'algorithm option: ["knn","mog2"]'

        if algorithm == 'MOG2':
            self.model = cv2.createBackgroundSubtractorMOG2()
        else:
            self.model = cv2.createBackgroundSubtractorKNN()
    
    def operate(self, img:np.ndarray, blur=True):
        if blur:
            img = cv2.GaussianBlur(img, (0,0), 1.5)
        
        return self.model.apply(img)


def make_STmap(prev, img, dim=0)->np.ndarray:
    assert dim >= 0 

    
    t_i = np.average(img, axis=dim, keepdims=True)
    
        
    return np.append(prev, t_i, axis=dim)
