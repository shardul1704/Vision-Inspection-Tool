import numpy as np
import cv2


img = cv2.imread('Test Images/compressor_green.png')


imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img, (5,5), 1)

v = np.median(imgBlur)
sigma = 0.33
    # ---- apply automatic Canny edge detection using the computed median----
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
imgCanny = cv2.Canny(imgBlur, lower, upper)
#cv2.imshow('Edges', edged)

#imgCanny = cv2.Canny(imgBlur,200,200)
kernel = np.ones((5,5))
imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
imgThres = cv2.erode(imgDial,kernel,iterations=1)
idx=0
biggest = np.array([])
maxArea = 0
contours, hierarchy = cv2.findContours(imgThres,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 5000:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if area > maxArea and len(approx) == 4:
            biggest = approx
            maxArea = area
            idx = cnt
# The index of the contour that surrounds your object
mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
out = np.zeros_like(img) # Extract out the object and place into output image
out[mask == 255] = img[mask == 255]

# Show the output image
cv2.imshow('Output', out)
cv2.waitKey(0)