import cv2
#print(cv2.__version__)
vidcap=cv2.VideoCapture('test.mp4')
success, img = vidcap.read()
count=0
success=True
while success:
	cv2.imwrite("C:/STUDYING/MACHINE LEARNING/Master/Topics/OpenCV/New/frame%d.jpg" % count,img)
	success,img=vidcap.read()
	print('Read a new frame:', success)
	count=count+1
print(count)