import os
import cv2
import subprocess as sp

paths = {
}

input_file = 'input_file_name.mp4'
output_file = 'output_file_name.mp4'

cap = cv2.VideoCapture('test.mp4')

cv2.namedWindow('Butterfly' ,cv2.WINDOW_NORMAL)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    else:
        break
cap.release()
cv2.destroyAllWindows()


#  gphoto2 --stdout --capture-movie | ffmpeg -re -i pipe:0 -listen 1 -f mjpeg http://localhost:8080/feed.jpg
# gphoto2 --stdout --capture-movie | ffmpeg -i - -pix_fmt yuv420p -threads 0 -f mp4 test.mp4