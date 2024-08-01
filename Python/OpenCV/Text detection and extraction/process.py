import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Read image from which text needs to be extracted
img = cv2.imread("sample.jpg")

# cv2.imshow('Doc',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Grayscale',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# cv2.imshow('ret',thresh1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Specify structure shape and kernel size. 
# Kernel size increases or decreases the area of the rectangle to be detected.
# A smaller value like (10, 10) will detect each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

# cv2.imshow('kernel',rect_kernel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

cv2.imshow('dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # # Finding contours
# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
# 												cv2.CHAIN_APPROX_NONE)


# cv2.imshow('contours',hierarchy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(hierarchy)


