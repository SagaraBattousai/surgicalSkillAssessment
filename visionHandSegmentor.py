import cv2
import numpy as np
import cvTools
from matchers import *

def segmentHand2(image_name, padding=None):

    image = cv2.imread(image_name)
    
    gloveCatcher = CyanColourMatcher()
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bestCont = gloveCatcher.getLargestContours(hsv)[0]

    tightBoundary = Boundary(cv2.convexHull(bestCont))

    x, y, w, h = cv2.boundingRect(cv2.convexHull(bestCont))

    # blank = np.zeros((*boundingBox.shape,1), np.uint8)
    # new_contors=[]

    
    # newHull = []

    # for hcx, hcy in hullCont:
    #     newHull.append(np.array((hcx - x, hcy - y)))

    # newHull = np.array(newHull)

    height, width, _ = image.shape

    blank = np.zeros((height, width, 1), dtype=np.uint8)
    tightBoundary.drawBoundary(blank, colour=255, width=-1)

    if padding is not None:
        tightBoundary.drawBoundary(blank, colour=255, width=padding)
        h += padding
        w += padding

    cvTools.displayImages(blank)

    pixelpoints = np.transpose(np.nonzero(blank))

    # print(pixelpoints.shape)
    # print(pixelpoints[0])
    # print(x, y)

    masked = cv2.bitwise_and(image, image, mask=blank)

    hand_image = np.zeros((h, w, 3), np.uint8)

    # aa, _, cc = hullCont.shape

    # hullCont = hullCont.reshape(aa, cc)

    for hch, hcw , _ in pixelpoints:
        hand_image[hch - y, hcw - x] = image[hch, hcw]

    # print("area:", cv2.contourArea(boxCont),
    #       "perimeter", cv2.arcLength(boxCont, True),
    #       "p2", cv2.arcLength(boxCont, False), sep="\n")

    cvTools.displayImages(hand_image)
    #return tightBoundary.points.shape
    # cvTools.displayImages(masked, img_a, img_b, blank)
    #cv2.drawContours(img_b, hullCont, -1, (255,255,0), 3)

def segmentHand(image_name, skin=False, display=False):

    image = cv2.imread(image_name)
    
    gloveCatcher = SkinColourMatcher() if skin else CyanColourMatcher()
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bestCont = gloveCatcher.getLargestContours(hsv)[0]

    tightBoundary = Boundary(cv2.convexHull(bestCont))

    x, y, w, h = cv2.boundingRect(bestCont)

    boarderBoundary = Boundary.fromRect(x, y, w, h)

    height, width, _ = image.shape

    blank = np.zeros((height, width, 1), dtype=np.uint8)

    boarderBoundary.drawBoundary(blank, colour=255, width=-1)
    
    pixelpoints = np.transpose(np.nonzero(blank))

    masked = cv2.bitwise_and(image, image, mask=blank)

    hand_image = np.zeros((h+1, w+1, 3), np.uint8)


    for hch, hcw , _ in pixelpoints:
        hand_image[hch - y, hcw - x] = image[hch, hcw]

    if display:
        cvTools.displayImages(hand_image)

    return {"topLeftX":x, "topLeftY":y, "width":w, "height":h}

def segmentDualHands(img):#, num=2):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gloveCatcher = CyanColourMatcher()

    contours = gloveCatcher.getLargestContours(hsv, 2)#num)

    convex = img.copy()

    box1 = Boundary(contours[0])
        
    box2 = Boundary(contours[1])

    box1.toRect()
        
    box2.toRect()

    box1.drawBoundary(convex)
        
    box2.drawBoundary(convex)

    return convex


if __name__ == "__main__":

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        cap.open()

    widthSet = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    heightSet = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    ret, frame = cap.read()
    
    frame_inputs = {}

    frame_count = 4617

    while(ret):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        gloveCatcher = CyanColourMatcher()

        contours = gloveCatcher.getLargestContours(hsv, 2)

        convex = frame.copy()

        box1 = Boundary(contours[0])
        
        box2 = Boundary(contours[1])

        box1.toRect()
        
        box2.toRect()

        box1.drawBoundary(convex)
        
        box2.drawBoundary(convex)

        frame_inputs["frame_" + str(frame_count)] = np.array(frame)

        print(str(frame_count), end="\r")

        cv2.imshow('bound', convex)

        frame_count += 1
    
        k = cv2.waitKey(25) & 0xFF
        if k == ord('q'):
            break
    
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    np.savez_compressed("DualHandData", **frame_inputs)

    # np.savez_compressed("Gloved_Hand_Labels_skin", **box_outputs)





