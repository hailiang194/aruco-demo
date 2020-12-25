import numpy as np
import cv2
import time

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
for i in range(255):
    for j in range(100):
        if np.array_equal(dictionary.bytesList[j][0][:1],[i]):
            print(dictionary.bytesList[j])

    print()
