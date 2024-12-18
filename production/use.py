import cv2

img = cv2.imread('Subject_0_0.png', cv2.IMREAD_GRAYSCALE) # 0
cv2.imshow('Window Title', cv2.resize(img, (300, 200)))
cv2.waitKey(0) # time to wait before closing in milliseconds (0 for infinite) 