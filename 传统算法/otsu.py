
import cv2
import numpy as np
from matplotlib import pyplot as plt

def otsu(path="./1.png"):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    SavePath = "./result.png"
    cv2.imwrite(SavePath,th1)
    return SavePath
    # plt.imshow(th1, "gray")
    # plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
    # plt.show()

if __name__ == '__main__':
    otsu()