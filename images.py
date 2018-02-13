import numpy as np
import cv2

imgG = cv2.imread('LennaComputerVision.png', 0)

cv2.imwrite("LennaGrey.png", imgG)


cap = cv2.VideoCapture("VideoSequence.avi")


fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 0)

        out.write(frame)

cap.release()
out.release()
