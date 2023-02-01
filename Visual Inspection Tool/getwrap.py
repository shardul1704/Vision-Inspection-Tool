import cv2
import numpy as np

widthImg, heightImg = 640,480
img = cv2.imread('compressor_green.png')
pts1 = np.float32([[48,22],[865,30],[63,686],[899,646]])
pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
cv2.imshow('warped',imgOutput)
cv2.imwrite('compressor_warped.jpg',imgOutput)
cv2.waitKey(0)


# https://ibb.co/sHcvcc9
# https://ibb.co/Ln7NFgd
# https://ibb.co/C2jx0M7