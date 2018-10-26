
import functools
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

inputPath = '/media/sf_vboxshare/testdata/'
inputs = ['Capture.PNG', 'Capture2.PNG']

NUM_CELLS_Y = 12

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def show_image(img, index, total):
    plt.subplot(100 + total * 10 + index + 1)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

def show_results(images):
    for i, image in enumerate(images):
        show_image(image, i, len(images))
    mng = plt.get_current_fig_manager()
    # mng.frame.Maximize(True)
    mng.resize(*mng.window.maxsize())
    plt.show()

def to_gray(input):
    img = cv2.imread(inputPath + input, cv2.IMREAD_COLOR)
    return (img, cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

def get_rectangles(inputTuple):
    (img, gray) = inputTuple
    _, result = cv2.threshold(gray, 0,255,0)
    ret,thresh = cv2.threshold(gray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    dst = cv2.drawContours(gray, contours, -1, (0,255,0), 2)
    #print(hierarchy, len(hierarchy[0]), len(contours))
    for i, cnt in enumerate(contours):
        epsilon = 0.03*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if True or approx.size == 4:
            area = cv2.contourArea(cnt);
            if hierarchy[0][i][2] > -1 and area > (200*200) and area < (600*600):
                #print(hierarchy[0][i][2])
                #dst = cv2.drawContours(result, [approx], -1, (0,0,255), 1)
                #print(approx, approx.size)
                leftmost = np.amin(approx[:,:,0])
                rightmost = np.amax(approx[:,:,0])
                topmost = np.amin(approx[:,:,1])
                bottommost = np.amax(approx[:,:,1])
                width = rightmost - leftmost
                height = bottommost - topmost
                aspect = width / height
                is12 = aspect < 1.7
                color = (0,255,0) if is12 else (255,0,0)
                dst = cv2.rectangle(img, (leftmost, topmost), (rightmost, bottommost), color, 2)
                divisor = 12 if is12 else 16
                cellWidth = width / divisor
                cellHeight = height / NUM_CELLS_Y
                for i in range(1, divisor):
                    x = int(leftmost + i * cellWidth)
                    dst = cv2.line(dst, (x, topmost), (x, bottommost), color, 1);
                for i in range(1, NUM_CELLS_Y):
                    y = int(topmost + i * cellHeight)
                    dst = cv2.line(dst, (leftmost, y), (rightmost, y), color, 1);
    return dst

process = compose(get_rectangles, to_gray)

results = [process(input) for input in inputs]
show_results(results)
