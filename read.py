
import functools
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

inputPath = '/media/sf_vboxshare/testdata/'
inputs = [ \
'Capture.PNG', 'Capture2.PNG',
'Capture3.PNG', 'Capture4.PNG' \
]

NUM_CELLS_Y = 12

# approximate grid dimensions compared to page dimensions
GRID_HEIGHT = 223/1120
GRID_WIDTH = 447/813

BAD_ANGLE_MAX = 3

data = {'patterns': []}

def formatData(data):
    for i, pattern in enumerate(data['patterns']):
        print('\nPattern ' + str(i))
        for j, track in enumerate(data['patterns'][i]['tracks']):
            print(str(j) + ': ' + str([x['value'] for x in track['values']]))

def area_minmax(pageSize):
    min = GRID_WIDTH * 0.5 * pageSize[0] * GRID_HEIGHT * 0.8 * pageSize[1]
    max = GRID_WIDTH * 1.2 * pageSize[0] * GRID_HEIGHT * 1.2 * pageSize[1]
    return min, max

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

def deltas(pt1, pt0):
    dx = float(pt1[0] - pt0[0])
    dy = float(pt1[1] - pt0[1])
    return np.abs(dx), np.abs(dy)

def get_approx_rect(approx, imgWidth, imgHeight):
    left = np.amin(approx[:,:,0])
    right = np.amax(approx[:,:,0])
    top = np.amin(approx[:,:,1])
    bottom = np.amax(approx[:,:,1])
    skewLimit = imgHeight * GRID_HEIGHT / 20
    # special handling to get rid of bad corners
    if len(approx) == 4:
        skewed = [deltas(approx[(i-1)%4][0], approx[i][0]) for i in range(0,4)]
        good = [[], []]
        for i in range(0, 4):
            skew = skewed[i]
            min = np.argmin(skew)
            max = np.argmax(skew)
            if skew[min] < skewLimit:
                good[max].append(approx[i][0][max])
                good[max].append(approx[(i-1)%4][0][max])
        left = np.amin(good[0])
        right = np.amax(good[0])
        top = np.amin(good[1])
        bottom = np.amax(good[1])

    return left, right, top, bottom

def get_rectangles(inputTuple):
    img, gray = inputTuple
    imgHeight, imgWidth, _ = img.shape
    areaMin, areaMax = area_minmax((imgWidth, imgHeight))
    _, result = cv2.threshold(gray, 0,255,0)
    ret,thresh = cv2.threshold(gray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # dst = cv2.drawContours(img, contours, -1, (0,255,0), 2)
    #print(hierarchy, len(hierarchy[0]), len(contours))
    for i, cnt in enumerate(contours):
        epsilon = 0.03*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if True or len(approx) == 4: # can't trust that approx is perfectly 4 cornered
            area = cv2.contourArea(cnt);
            if hierarchy[0][i][2] > -1 and area > areaMin and area < areaMax:
                #print(hierarchy[0][i][2])
                # dst = cv2.drawContours(img, [approx], -1, (0,128,200), 1)
                #print(approx, len(approx))
                left, right, top, bottom = get_approx_rect(approx, imgWidth, imgHeight)
                width = right - left
                height = bottom - top
                aspect = width / height
                is12 = aspect < 1.7
                color = (0,255,0) if is12 else (255,0,0)
                dst = cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                divisor = 12 if is12 else 16
                cellWidth = width / divisor
                cellHeight = height / NUM_CELLS_Y
                trackData = {'tracks': []}
                data['patterns'].append(trackData)
                for j in range(0, NUM_CELLS_Y):
                    trackData['tracks'].append({'values': []})
                    values = trackData['tracks'][j]['values']
                    y = int(top + j * cellHeight)
                    dst = cv2.line(dst, (left, y), (right, y), color, 1);
                    for i in range(0, divisor):
                        x = int(left + i * cellWidth)
                        dst = cv2.line(dst, (x, top), (x, bottom), color, 1);
                        center = (int(x + cellWidth/2), int(y + cellHeight/2))
                        #print(center, x, y, top, left, bottom, right)
                        value = 0 if gray[center[1], center[0]] > 128 else 1
                        if value:
                            dst = cv2.line(dst, (center[0], center[1]), (center[0]+2, center[1]+2), color, 3)
                        values.append({'value': value})
    return dst

process = compose(get_rectangles, to_gray)

result = [process(input) for input in inputs]
formatData(data)
show_results(result)
