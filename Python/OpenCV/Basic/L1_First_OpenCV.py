# Import OpenCV library
import cv2
# Open image
img=cv2.imread('02.jpg',0)
# Show image
cv2.imshow('image', img)
# Wait to click close image
cv2.waitKey(0)
cv2.destroyAllWindows()