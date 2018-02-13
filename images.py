import numpy as np
import cv2


def main():

    imgG = cv2.imread('LennaComputerVision.png', 0)

    cv2.imwrite("LennaGrey.png", imgG)


    cap = cv2.VideoCapture("VideoSequence.avi")


    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter('man.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


    i = 100

    while(cap.isOpened() and i >= 0):
        ret, frame = cap.read()

        if ret:
            frame = cv2.flip(frame, 0)

            out.write(frame)
        i -= 1

    cap.release()
    out.release()


main()
