from skimage.measure import compare_ssim
import cv2
import numpy as np

original = cv2.imread('Images/original.png')
test = cv2.imread('Images/test1.png')
test2 = cv2.imread('Images/test2.png')
test3 = cv2.imread('Images/test3.png')


original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
test2_gray = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)
test3_gray = cv2.cvtColor(test3, cv2.COLOR_BGR2GRAY)


(score, diff) = compare_ssim(original_gray, test_gray, full=True)
(score2, diff2) = compare_ssim(original_gray, test2_gray, full=True)
(score3, diff3) = compare_ssim(original_gray, test3_gray, full=True)
print("Image similarity", score)
print("Image similarity", score2)
print("Image similarity", score3)


diff = (diff * 255).astype("uint8")
diff2 = (diff2 * 255).astype("uint8")
diff3 = (diff3 * 255).astype("uint8")


thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh2 = cv2.threshold(diff2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh3 = cv2.threshold(diff3, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
contours2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2 = contours2[0] if len(contours2) == 2 else contours2[1]
contours3 = cv2.findContours(thresh3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours3 = contours3[0] if len(contours3) == 2 else contours3[1]

for c in contours:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        #cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(test, (x, y), (x + w, y + h), (36,255,12), 2)

for c2 in contours2:
    area = cv2.contourArea(c2)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c2)
        #cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(test2, (x, y), (x + w, y + h), (36,255,12), 2)

for c3 in contours:
    area = cv2.contourArea(c3)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c3)
        #cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(test3, (x, y), (x + w, y + h), (36,255,12), 2)

cv2.imshow('before', original)
cv2.imshow('after', test)
cv2.imshow('after2', test2)
cv2.imshow('after3', test3)
#cv2.imshow('diff',diff)

cv2.waitKey(0)