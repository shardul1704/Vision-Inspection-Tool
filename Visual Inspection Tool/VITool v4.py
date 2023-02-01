import cv2
import numpy as np
from skimage.measure import compare_ssim

#################################
#widthImg, heightImg = 480, 640
widthImg, heightImg = 640, 480
#################################


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)

    v = np.median(imgBlur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    imgCanny = cv2.Canny(imgBlur, lower, upper)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres


def getContours(img,imgContour):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1,(255,0,0),20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return imgOutput

def Warped(cap):
    img = cap.copy()
    img = cv2.resize(img, (widthImg,heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres,imgContour)
    if biggest.size!=0:
        imgWarped = getWarp(img,biggest)
        imageArray = ([img, imgThres],
                    [imgContour,imgWarped])
    else:
        imageArray = ([imgContour, img])
        print("Biggest Contour could not be found!")

    stackedImages = stackImages(0.7,imageArray)
    cv2.imshow("workflow",stackedImages)
    return imgWarped


def getDifference(original, test):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(original_gray, test_gray, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
    drawBasicGrid(original)
    drawBasicGrid(test)
    drawBasicGrid(diff)
    finalstack = stackImages(0.7,[original,test,diff])
    cv2.imshow('Result', finalstack)
    #cv2.imshow('after', test)
    #cv2.imshow('diff', diff)


def drawBasicGrid(img):
    x,y=0,0
    # Draw all x lines
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=(255, 0, 255), thickness=1)
        x += img.shape[1]//3

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=(255, 0, 255), thickness=1)
        y += img.shape[0]//3


'''
original1 = cv2.imread('Test Images/std test 1/original1.jpg')
path = 'Test Images/std test 1/'
variant = input("Please Enter the Variant Number (1,2,3,5,6,7,8) = ")
path = path + variant + '.jpg'
print(path)

test1 = cv2.imread(path)

original_warped = Warped(original1)
test_warped = Warped(test1)
getDifference(original_warped,test_warped)

cv2.waitKey(0)'''


variant = input("Please enter the variant number =")
original_path = 'Test Images/'+variant+'/original.png'
original1 = cv2.imread(original_path)
testimg_path = 'Test Images/'+variant+'/'+variant+'.png'
test1 = cv2.imread(testimg_path)
# if original1 == None:
#     original_path = 'Test Images/'+variant+'/original.jpg'
#     original1 = cv2.imread(original_path)
#     testimg_path = 'Test Images/'+variant+'/'+variant+'.jpg'
#     test1 = cv2.imread(testimg_path)
original_warped = Warped(original1)
test_warped = Warped(test1)
getDifference(original_warped,test_warped)

cv2.waitKey(0)

#     https://github.com/raj713335/FACIAL_RECOGNITION

# CONSTRAINTS:
#
# 1.	Works accurately on HD images.
# 2.	Only works when there is very less background Noise.
# 3.	The biggest contour found has to be a square/rectangle to get the warp perspective.
# 4.	Finding contours usually is not accurate.
