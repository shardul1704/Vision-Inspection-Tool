import cv2
import numpy as np
import imutils

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
    imgBlur = cv2.GaussianBlur(img, (5,5), 1)

    v = np.median(imgBlur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    imgCanny = cv2.Canny(imgBlur, lower, upper)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres


def templateMatch(img,test):
    testing = preProcessing(test)
    gray = preProcessing(img)
    (tH, tW) = testing.shape[:2]
    found = None
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = resized.copy()
        result = cv2.matchTemplate(edged, testing, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # draw a bounding box around the detected result and display the image
    #cv2.rectangle(original_image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return img
    #cv2.imshow("Image", original_image)
    #cv2.waitKey(0)


original_image = cv2.imread('compressor_template.png')
#original_image = cv2.imread('cod_mw.jpg')
(oH, oW) = original_image.shape[:2]
#roi1 = original_image[0:(oH//2),0:(oW//2)]
roi1 = original_image[0:(2*oH//3),0:(2*oW//3)]
roi2 = original_image[(oH//2)-90:oH,(oW//2)-50:oW]
cv2.imshow("roi1",roi1)
cv2.imshow("roi2",roi2)
test1 = cv2.imread('test compressor/atlas logo.png')
test2 = cv2.imread('test compressor/screen.png')
test3 = cv2.imread('test compressor/vent.png')
test4 = cv2.imread('cod_logo.png')
mask = preProcessing(original_image)
cv2.imshow('mask',mask)
res = templateMatch(roi1,test1)
templateMatch(roi1,test2)
templateMatch(roi1,test3)
#templateMatch(original_image,test4)
#cv2.imshow("Image", original_image)
cv2.imshow("Image", res)
cv2.waitKey(0)

