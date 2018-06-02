import cv2
import numpy as np
from datetime import datetime

#101 104 101
#Key was to make sat from 50 to 255 instead of 100

class ColourMatcher:
    def __init__(self, upper_bound, lower_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def maskColour(self, hsv):
        return cv2.inRange(hsv, self.lower_bound, self.upper_bound)

class CyanColourMatcher(ColourMatcher):
    def __init__(self):
        upper_cyan = np.array([110, 255, 255])
        lower_cyan = np.array([90, 75, 75])

        super().__init__(upper_cyan, lower_cyan)

class SkinColourMatcher(ColourMatcher):
    def __init__(self):
        upper_cyan = np.array([20, 255, 255])
        lower_cyan = np.array([0, 75, 75])

        super().__init__(upper_cyan, lower_cyan)

class BoundingBox:
    def __init__(self, centers, bounds, angle):
        self.centerX = centers[0]
        self.centerY = centers[1]
        self.width = bounds[0]
        self.height = bounds[1]
        self.angle = angle

    def getAsTuple(self):
        return ((self.centerX, self.centerY), (self.width, self.height), self.angle)

    def toBox(self):
        box = cv2.boxPoints(self.getAsTuple())
        return np.int0(box)

    def getArea(self):
        return self.width * self.height

    def __eq__(self, other):
        if isinstance(other, BoundingBox):
            return self.centerX == other.centerX and \
                   self.centerY == other.centerY and \
                   self.width   == other.width   and \
                   self.height  == other.height  and \
                   self.angle   == other.angle  


        #Safe Due to short circuiting
        elif isinstance(other, tuple) and len(other) == 3:
            try:
                return self.centerX == other[0][0] and \
                       self.centerY == other[0][1] and \
                       self.width   == other[1][0] and \
                       self.height  == other[1][1] and \
                       self.angle   == other[2]
            except IndexError:
                return NotImplemented

        return NotImplemented

    def __gt__(self, other):
        
        if isinstance(other, BoundingBox):
            return self.width *  self.height > other.width * other.height

        #Safe Due to short circuiting
        elif isinstance(other, tuple) and len(other) == 3:
            try:
                return self.width * self.height > other[1][0] * other[1][1]
            except IndexError:
                return NotImplemented

        return NotImplemented


def showCentroid(img, contour, colour=[255, 0, 255]):

    M = cv2.moments(contour)

    try:
        cx = int(M['m10'] / M['m00'])
    except ZeroDivisionError:
        return
    try:
        cy = int(M['m01'] / M['m00'])
    except ZeroDivisionError:
        return

    height, width, _ = img.shape

    if (cx + 10) > width or (cy + 10) > height:
        return

    xStart = cx - 10
    yStart = cy - 10
    for i in range(20):
        for j in range(20):
            img[yStart + i][xStart + j] = colour


dt = datetime.now()

dtm = str(dt.minute) if dt.minute > 9 else "0" + str(dt.minute)

date_format = "_{}_{}:{}".format(dt.date(), dt.hour, dtm)

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    cap.open()

ret, frame = cap.read()
    
frame_inputs = {}

box_outputs = {}

frame_count = 0

while(ret):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gloveCatcher = SkinColourMatcher()#CyanColourMatcher()

    mask = gloveCatcher.maskColour(hsv)

    origional_mask, cont, heir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    convex = frame.copy()

    bb = BoundingBox((0,0),(0,0),0)


    for cnt in cont:
        possible_bb = BoundingBox(*cv2.minAreaRect(cnt))
        if possible_bb > bb:
            bb = possible_bb
           
    if bb != ((0,0),(0,0),0):
        box = bb.toBox()
        cv2.drawContours(convex, [box], 0, (0, 255, 255), 5)
    
    frame_count += 1

    frame_inputs["frame_" + str(frame_count)] = np.array(frame)
    box_outputs["frame_" + str(frame_count)] = np.array(box)

    print(str(frame_count), end="\r")

    cv2.imshow('frame', frame)
    cv2.imshow('bound', convex)
    
    k = cv2.waitKey(25) & 0xFF
    if k == ord('q'):
        break
    
    ret, frame = cap.read()

cv2.destroyAllWindows()

np.savez_compressed("Gloved_Hand_Training_Data_skin", **frame_inputs)

np.savez_compressed("Gloved_Hand_Labels_skin", **box_outputs)





