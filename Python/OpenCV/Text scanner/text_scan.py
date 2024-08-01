# Step 1: Read input image
import cv2

# file name
input_image  = 'mydoc.jpg'

# read image
image = cv2.imread(input_image)

# show image
# cv2.imshow('Input', image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Step 2: detect image's edge
# Convert color into gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blurring gray image to decrease noise
blur = cv2.blur(gray,(3,3))

# edge detect using Canny method
edge = cv2.Canny(blur, 50, 300, 3)

# Show images
# cv2.imshow("gray", gray)
# cv2.imshow("blur", blur)
# cv2.imshow("edge", edge)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Step 3: Get image's edge

# Find contours
cnts = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
import imutils
cnts = imutils.grab_contours(cnts)

# Sorting contours according to descending area
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Get first contour - biggest
cnts = cnts[:1]

# Draw contour into original image to visualize
p = cv2.arcLength(cnts[0], True)
r = cv2.approxPolyDP(cnts[0], 0.02*p, True)
# cv2.drawContours(image, [r], -1, (0,0,255), 3)

# Show image
# cv2.imshow("Draw", image)
# cv2.waitKey()

# Step 4: Rotate image to straight
# Đầu tiên reshape cái ROI của chúng ta về (4,2) - 4 tọa độ, mỗi tọa độ gồm x,y
r = r.reshape(4,2)

# Tính toán 04 góc theo thứ tự trên trái, trên phải, dưới phải, dưới trái
import numpy as np
rect = np.zeros((4,2), dtype='float32')

# Ta tính tổng các tọa độ theo cột
# Điểm trên trái sẽ có tổng nhỏ nhất
# Điểm dưới phải sẽ có tổng lớn nhất
s = np.sum(r, axis=1)
rect[0] = r[np.argmin(s)] # Trên trái
rect[2] = r[np.argmax(s)] # Dưới phải

# Ta tính sự khác nhau giữa các tọa độ theo cột
# Trên phải sẽ ít khác biệt nhất
# dưới trái là khác biệt nhất
diff = np.diff(r, axis=1)
rect[1] = r[np.argmin(diff)]
rect[3] = r[np.argmax(diff)]

# Tính toán chiều rộng và chiều cao của văn bản
(tl, tr, br, bl) = rect

width1 = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
width2 = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
Width = max(int(width1), int(width2))

height1 = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
height2 = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
Height = max(int(height1), int(height2))

# Rotate image:
# Tọa độ mới của văn bản
new_rect = np.array([
    [0,0],
    [Width-1, 0],
    [Width-1, Height-1],
    [0, Height-1]], dtype="float32")

# Tinh toán ma trận transform
M = cv2.getPerspectiveTransform(rect, new_rect)

# Thực hiện xoay và crop
output = cv2.warpPerspective(image, M, (Width, Height))

# Show ảnh
# cv2.imshow("Output",output)
# cv2.waitKey()

# Step 5: Result processing:
# Convert into gray
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

# Áp threshold

_, output_final = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

# Show hàng
cv2.imshow("Ouput", output_final)
cv2.waitKey()

