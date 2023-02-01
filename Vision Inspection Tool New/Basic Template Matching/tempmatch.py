import imutils
import cv2
import numpy as np
import time

def preProcessing(img):
   imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

   v = np.median(imgBlur)
   sigma = 0.33
   lower = int(max(0, (1.0 - sigma) * v))
   upper = int(min(255, (1.0 + sigma) * v))
   imgCanny = cv2.Canny(imgBlur, lower, upper)
   kernel = np.ones((5, 5))
   imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
   imgThresh = cv2.erode(imgDial, kernel, iterations=1)
   return imgThresh


def templateMatch(img, test):
   testing = preProcessing(test)
   gray = preProcessing(img)
   (tH, tW) = testing.shape[:2]
   found = None
   for scale in np.linspace(0.2, 1.0, 20)[::-1]:
      resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
      r = gray.shape[1] / float(resized.shape[1])
      if resized.shape[0] < tH or resized.shape[1] < tW:
         break
      edged = cv2.Canny(resized, 50, 200)
      result = cv2.matchTemplate(edged, testing, cv2.TM_CCOEFF)
      (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
      clone = np.dstack([edged, edged, edged])
      cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
      cv2.imshow("Visualize", clone)
      if found is None or maxVal > found[0]:
         found = (maxVal, maxLoc, r)
   (_, maxLoc, r) = found
   (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
   (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
   cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
   cv2.putText(img, 'Correct', (startX - 2, startY - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
   #img2 = cv2.resize(img,(640,480))
   cv2.imshow("final",img)


main_image = cv2.imread('Test Images/Compressor/C3.jpg')
#cv2.imshow("template",template)
template = cv2.imread('Test Images/Template/T5.JPG')
cv2.imshow("template",template)
#cv2.imshow("main",main_image)
templateMatch(main_image,template)

cv2.waitKey(0)