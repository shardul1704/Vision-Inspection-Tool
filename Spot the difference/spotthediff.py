from skimage.measure import compare_ssim
import cv2
import numpy as np

original = cv2.imread('Images/result.png')
test = cv2.imread('Images/result1.png')


original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)


(score, diff) = compare_ssim(original_gray, test_gray, full=True)
print("Image similarity", score)


diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]


for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x,y,w,h = cv2.boundingRect(c)

        cv2.rectangle(test, (x, y), (x + w, y + h), (36,255,12), 2)

cv2.imshow('before', original)
cv2.imshow('after', test)
cv2.imshow('diff',diff)

cv2.waitKey(0)