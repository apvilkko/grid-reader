import functools
import cv2
import numpy as np
from matplotlib import pyplot as plt

# inputPath = '/media/sf_vboxshare/testdata/'
inputPath = 'testdata/'
outputPath = 'output/'
# inputs = [
#     'Capture.PNG',
#     'Capture2.PNG',
#     'Capture3.PNG',
#     'Capture4.PNG',
#     'Capture5.PNG'
# ]
TEST_FLAM1 = 15
TEST_FLAM2 = 44
TEST_GRID1 = 18
TEST_GRID2 = 19
TEST_GRID3 = 20
TEST_GRID4 = 23
TEST_GRID5_TAM = 26
TEST_GRID6 = 27
TEST_GRID7 = 38
TEST_GRID8 = 49
# inputPages = range(8, 98)
inputPages = [
    # TEST_GRID1,
    # TEST_GRID2,
    TEST_GRID3, TEST_GRID4
]
inputs = ['page-{:02d}.png'.format(x) for x in inputPages]

NUM_CELLS_Y = 12

# approximate grid dimensions compared to page dimensions
GRID_HEIGHT = 223/1120
GRID_WIDTH = 447/813

BAD_ANGLE_MAX = 3

VALUE_ON = 1
VALUE_OFF = 0
VALUE_FLAM = 2

DEFAULTS = ['AC', 'RD', 'CH', 'OH', 'HT', 'MT', 'SD', 'RS', 'LT', 'CP', 'CB', 'BD']
CY = 1
TAM = 10

gData = {'patterns': []}

debugBlocks = []


def resize_image_viewer():
    mng = plt.get_current_fig_manager()
    # mng.frame.Maximize(True)
    maxSize = mng.window.maxsize()
    maxSize = (maxSize[0]*0.5, maxSize[1])
    mng.resize(*maxSize)
    mng.window.wm_geometry('-1000-0')


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
    plt.imshow(img, cmap='gray', interpolation='bicubic')


def show_results(images):
    for i, image in enumerate(images):
        show_image(image, i, len(images))
    resize_image_viewer()
    plt.show()


def write_outputs(images):
    for i, image in enumerate(images):
        cv2.imwrite('{}out{:02d}.png'.format(outputPath, i), image)


def read_image(input):
    return cv2.imread(inputPath + input, cv2.IMREAD_COLOR)


def to_gray(input):
    return (input, cv2.cvtColor(input, cv2.COLOR_BGR2GRAY))


def deltas(pt1, pt0):
    dx = float(pt1[0] - pt0[0])
    dy = float(pt1[1] - pt0[1])
    return np.abs(dx), np.abs(dy)


def distance(pt1, pt0):
    return np.sqrt((pt1[0]-pt0[0])**2+(pt1[1]-pt0[1])**2)


def ri(x):
    return int(round(x))


def get_approx_rect(approx, imgWidth, imgHeight):
    left = np.amin(approx[:, :, 0])
    right = np.amax(approx[:, :, 0])
    top = np.amin(approx[:, :, 1])
    bottom = np.amax(approx[:, :, 1])
    skewLimit = imgHeight * GRID_HEIGHT * 0.02
    shrink = imgHeight * GRID_HEIGHT * 0.01
    minPathLen = imgHeight * GRID_HEIGHT * 0.05
    # special handling to get rid of bad corners
    dim = len(approx)
    print('dim', dim)
    skewed = [deltas(approx[(i-1) % dim][0], approx[i][0]) for i in range(0, dim)]
    dist = [distance(approx[(i-1) % dim][0], approx[i][0]) for i in range(0, dim)]
    bins = 128
    topBins = ri(bins/8)
    histX = np.histogram(approx[:, :, 0], bins)
    histY = np.histogram(approx[:, :, 1], bins)
    left = ri(histX[1][np.argmax(histX[0][:topBins])])
    top = ri(histY[1][np.argmax(histY[0][:topBins])])
    gridHeight = GRID_HEIGHT * imgHeight

    # plt.hist(approx[:, :, 0], bins)
    # resize_image_viewer()
    # plt.show()

    print('topBins', histX[0][:topBins])
    print('most likely left', left)
    print('most likely top', top)
    print('most likely bottom', top + gridHeight)
    good = [[], []]
    mins = [np.argmin(skewed[i]) for i in range(0, dim)]
    for i in range(0, dim):
        skew = skewed[i]
        max = np.argmax(skew)
        min = mins[i]
        print('skew', skew, skew[min], approx[i], skewLimit, dist[i])
        longEnough = dist[i] > minPathLen
        val1 = approx[i][0][max]
        val2 = approx[(i-1) % dim][0][max]
        if skew[min] < skewLimit and longEnough and (
            (max == 1 and val1 >= top and val2 >= top) or (
            max == 0 and val1 >= left and val2 >= top)):
            good[max].append(val1)
            good[max].append(val2)
    if len(good[0]) and len(good[1]):
        print('good', good)
        # left = ri(np.amin(good[0]) + shrink)
        right = ri(np.amax(good[0]) - shrink)
        # top = ri(np.amin(good[1]) + shrink)
        bottom = ri(np.amax(good[1]) - shrink*2)
        delta = bottom - top
        if gridHeight - delta > gridHeight/10:
            bottom += ri(gridHeight - delta)
        elif delta - gridHeight > gridHeight/10:
            bottom -= ri(delta - gridHeight)
    else:
        print('no good corners')
        for i in range(0, dim):
            min = mins[i]
            skew = skewed[i][min]
            print('  ', approx[i], min, skew)

    return left, right, top, bottom


def has_flam(center, idx, yidx, gray, cellWidth, cellHeight, previousHasValue):
    # if idx == 0:
    #    # First cell needs special handling since flam marker is outside the grid
    #    return False
    if previousHasValue:
        # needs to be handled separately
        return False
    SECTION = (0.5, 1.1, 0.25, 0.75)  # relative section to crop for detecting flam
    if idx > 0:
        prevCenter = (center[0] - cellWidth, center[1])
        x = prevCenter[0]
        y = prevCenter[1]
        left = ri(x + cellWidth*SECTION[0] - cellWidth/2)
        right = ri(x + cellWidth*SECTION[1] - cellWidth/2)
        top = ri(y + cellHeight*SECTION[2] - cellHeight/2)
        bottom = ri(y + cellHeight*SECTION[3] - cellHeight/2)
        step = ri(cellHeight*0.2)
        if gray[top, ri(x)] < 128:
            # print('adjusting down', gray[top, ri(x)])
            top += step
            bottom += step
        rightEdgeMean = np.mean(gray[top:bottom, right-1:right].reshape(-1))
        if rightEdgeMean < 64:
            # print('adjusting left', rightEdgeMean)
            left -= step
            right -= step
        prevBlock = gray[top:bottom, left:right]
        rows, cols = prevBlock.shape
        kernelBlock = prevBlock[ri(rows*0.4):ri(rows*0.6), ri(cols*0.4):ri(cols*0.6)]
        # rightBlock = prevBlock[ri(rows*0.1):ri(rows*0.9), ri(cols*0.5):cols]
        mean = np.mean(prevBlock.reshape(-1))
        variance = np.var(prevBlock.reshape(-1))
        kernelVariance = np.var(kernelBlock.reshape(-1))
        kernelMean = np.mean(kernelBlock.reshape(-1))
        kernelCheck = kernelVariance > 30 and kernelMean < 240
        isFlam = variance > 100 and mean < 240 and kernelCheck
        # print(mean[0], len(debugBlocks))
        if isFlam:
            # debugBlocks.append(prevBlock)
            # print('flam', idx, yidx, variance, mean, kernelVariance, kernelMean)
            return True


def get_instruments(gray, left, right, top, bottom, cellWidth, cellHeight):
    instruments = DEFAULTS[:]
    cyBlock = gray[
        ri(top + CY*cellHeight):ri(top + (CY+1)*cellHeight),
        ri(left - 3*cellWidth):ri(left-0.3*cellWidth)]
    _, thresh = cv2.threshold(cyBlock, 127, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areaMin = (cellHeight * 0.3)**2
    areaMax = (cellHeight*0.53)**2
    contours = [x for x in contours if areaMin < cv2.contourArea(x) < areaMax]
    # for i, cnt in enumerate(contours):
    #    area = cv2.contourArea(cnt)
    #    print('area', area, areaMin, areaMax)
    # cyBlock = cv2.drawContours(cyBlock, contours, -1, (128, 128, 128), 1)
    if len(contours) == 3:
        # debugBlocks.append(cyBlock)
        instruments[CY] = 'CR'
        # print('cyBlock', mean, variance)
    yShift = ri(cellHeight*0.2)
    tamBlock = gray[
        ri(top + yShift + TAM*cellHeight):ri(top + yShift + (TAM+1)*cellHeight),
        ri(left - 3*cellWidth):ri(left-0.3*cellWidth)]
    _, thresh = cv2.threshold(tamBlock, 127, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areaMin = (cellHeight * 0.2)**2
    areaMax = cellHeight**2
    contours = [x for x in contours if areaMin < cv2.contourArea(x) < areaMax]
    # tamBlock = cv2.drawContours(tamBlock, contours, -1, (128, 128, 128), 1)
    if len(contours) > 2:
        instruments[TAM] = 'TA'
        debugBlocks.append(tamBlock)
    return instruments


def process_grid(approx, img, gray):
    imgHeight, imgWidth, _ = img.shape
    # print(hierarchy[0][i][2])
    # dst = cv2.drawContours(img, [approx], -1, (0,128,200), 1)
    # print(approx, len(approx))
    left, right, top, bottom = get_approx_rect(approx, imgWidth, imgHeight)
    width = right - left
    height = bottom - top
    aspect = width / height
    is12 = aspect < 1.7
    color = (0, 255, 0) if is12 else (255, 0, 0)
    dst = cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    divisor = 12 if is12 else 16
    cellWidth = width / divisor
    shiftLeft = ri(cellWidth / 6)
    cellHeight = height / NUM_CELLS_Y
    trackData = {'tracks': []}
    gData['patterns'].append(trackData)
    trackData['instruments'] = get_instruments(
        gray, left, right, top, bottom, cellWidth, cellHeight)
    cv2.putText(
        dst,
        str(trackData['instruments'][CY]) + ',' + str(trackData['instruments'][TAM]),
        (left - shiftLeft * 2, top - shiftLeft),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    for j in range(0, NUM_CELLS_Y):
        trackData['tracks'].append({'values': []})
        values = trackData['tracks'][j]['values']
        y = ri(top + j * cellHeight)
        dst = cv2.line(dst, (left, y), (right, y), color, 1)
        for i in range(0, divisor):
            x = ri(left + i * cellWidth)
            dst = cv2.line(dst, (x, top), (x, bottom), color, 1)
            center = (ri(x + cellWidth/2), ri(y + cellHeight/2))
            # print(center, x, y, top, left, bottom, right)
            # Shift left because flam could be detected as regular hit

            value = detect_value(gray, center, shiftLeft)
            previousHasValue = i > 0 and values[i - 1]['value'] != VALUE_OFF
            if value and has_flam(center, i, j, gray, cellWidth, cellHeight, previousHasValue):
                value = VALUE_FLAM
            if value == VALUE_ON:
                dst = cv2.line(dst, (center[0], center[1]), (center[0]+2, center[1]+2), color, 6)
            elif value == VALUE_FLAM:
                dst = cv2.line(dst, (
                    center[0]+2, center[1]), (center[0]+2+3, center[1]+3), (255, 255, 255), 6)
                dst = cv2.line(dst, (
                    center[0], center[1]), (center[0]+3, center[1]+3), (0, 0, 255), 6)
            values.append({'value': value})
    return dst


def detect_value(gray, center, shift):
    ret = VALUE_OFF
    val = np.median(gray[
        ri(center[1]-shift/2):ri(center[1]+shift/2),
        ri(center[0]-shift):center[0],
    ])
    if val < 127:
        ret = VALUE_ON
    return ret


def angle(line):
    return np.arctan((line[1]-line[3])/(line[2]-line[0]+1e-15))*180/np.pi


def straighten(img):
    imgHeight, imgWidth, _ = img.shape
    minLineLen = ri(imgWidth/3)
    threshold = 18
    edges = cv2.Canny(img, 100, 200)
    theta = np.pi/180
    lines = cv2.HoughLinesP(edges, 1, theta, threshold, 0, minLineLen, 20)
    rotation = 0
    if (lines is None or len(lines) == 0):
        print('no lines!')
    else:
        angles = [angle(line[0]) for line in lines]
        angles = [x for x in angles if -3 < x < 3]
        rotation = np.median(angles)
        print('rotation', rotation)
    m = cv2.getRotationMatrix2D((imgWidth/2, imgHeight/2), -1 * rotation, 1)
    return cv2.warpAffine(img, m, (imgWidth, imgHeight))


def process_page(inputTuple):
    img, gray = inputTuple
    imgHeight, imgWidth, _ = img.shape
    areaMin, areaMax = area_minmax((imgWidth, imgHeight))
    # cv2.equalizeHist(gray)
    enhanced = gray
    # contrast = 1.2
    # brightness = 10
    # enhanced = cv2.addWeighted(gray, contrast, gray, 0, brightness)
    # kernel = np.ones((3,3),np.uint8)
    # enhanced = cv2.dilate(enhanced, kernel, 1)
    # _, result = cv2.threshold(gray, 0, 255, 0)
    ret, thresh = cv2.threshold(enhanced, 170, 255, 0)

    # kernel = np.ones((3,3),np.uint8)
    # thresh = cv2.erode(thresh,kernel,iterations = 1)
    # thresh = cv2.dilate(thresh,kernel,iterations = 1)

    # plt.imshow(thresh)
    # resize_image_viewer()
    # plt.show()
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy, len(hierarchy[0]), len(contours))
    dst = img

    # edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    # theta = np.pi/180
    # threshold = 18
    # minLineLen = ri(imgWidth / 3)
    # lines = cv2.HoughLinesP(edges, 1, theta, threshold, 0, minLineLen, 20)
    # a,b,c = lines.shape
    # for i in range(a):
    #     dst = cv2.line(edges, (
    #         lines[i][0][0], lines[i][0][1]), (
    #         lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    #
    # show_image(dst, 0, 1)
    # plt.show()

    # dst = cv2.drawContours(img, contours, -1, (0,255,0), 2)
    for i, cnt in enumerate(contours):
        # epsilon = 0.03*cv2.arcLength(cnt, True)
        # epsilon = GRID_WIDTH * imgWidth * 0.009
        epsilon = GRID_WIDTH * imgWidth * 0.02
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if True or len(approx) == 4:  # can't trust that approx is perfectly 4 cornered
            area = cv2.contourArea(cnt)
            areaSatisfied = area > areaMin and area < areaMax
            if areaSatisfied:
                dst = cv2.drawContours(dst, [approx], -1, (255,100,0), 3)
            if hierarchy[0][i][2] > -1 and areaSatisfied:
                dst = process_grid(approx, img, gray)
    return dst


process = compose(process_page, to_gray, straighten, read_image)

result = [process(input) for input in inputs]
# formatData(gData)
show_results(result)
# write_outputs(result)
# show_results(debugBlocks[:9])
