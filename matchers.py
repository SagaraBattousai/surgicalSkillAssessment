import numpy as np
import cv2
import utils

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

class Boundary:
    def __init__(self, points):
        self.points = points.reshape(-1,1,2)

    @staticmethod
    def rectToPoints(x, y, width, height):
        yEnd = y + height
        xEnd = x + width
        return np.array([[[x, y]],
                           [[x, yEnd]],
                           [[xEnd, yEnd]],
                           [[xEnd, y]]],
                           dtype=np.int64)

    @classmethod
    def fromRect(cls, x, y, width, height):

        points = Boundary.rectToPoints(x, y, width, height)

        rect = cls.__new__(cls)
        
        rect.__init__(points)

        return rect
    
    @classmethod
    def fromRotatedRect(cls, rr):
        points = cv2.boxPoints(rr).reshape(-1,1,2)

        cls.__init__(cls, points)

    def toRect(self):
        self.points = Boundary.rectToPoints(
                                *cv2.boundingRect(self.points))

    def getArea(self):
        return cv2.contourArea(self.points)
    
    def getPerimeter(self):
        return cv2.contourArea(self.points)

    def drawBoundary(self, image, colour=255, width=5, copy=False):

        img = image.copy() if copy else image

        return cv2.drawContours(img, [self.points], 0, colour, width)

    def __eq__(self, other):
        if isinstance(other, BoundingBox):
            equality = self.points == other.points
        
        elif isinstance(other, (tuple, list, np.ndarray)):
            equality = self.points == other

        else:
            return NotImplemented

        if isinstance(equality, bool):
            return equality
        return equality.all()


    def __gt__(self, other):
        
        if isinstance(other, BoundingBox):
            return self.getArea() > other.getArea()

        if isinstance(other, (tuple, list, np.ndarray)):
            try:
                pointArray = np.array(other).reshape(-1, 2)
                area = cv2.contourArea(pointArray)
                return self.getArea() > area
            except ValueError:
                return False

        return NotImplemented

class ColourMatcher:
    def __init__(self, upper_bound, lower_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def maskColour(self, hsv):
        return cv2.inRange(hsv, self.lower_bound, self.upper_bound)

    def getContours(self, hsv):

        mask, contours, heirachy = cv2.findContours(self.maskColour(hsv), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return mask, contours, heirachy

    def getLargestContours(self, hsv, number=1):
    
        _, contours, _ = self.getContours(hsv)

        indecies = utils.getMaxIndeciesWith(len, contours, number)

        return [contours[x] for x in indecies]

class CyanColourMatcher(ColourMatcher):
    def __init__(self):
        upper_cyan = np.array([110, 255, 255])
        lower_cyan = np.array([90, 85, 85])

        super().__init__(upper_cyan, lower_cyan)

class SkinColourMatcher(ColourMatcher):
    def __init__(self):
        upper_cyan = np.array([20, 255, 255])
        lower_cyan = np.array([0, 75, 75])

        super().__init__(upper_cyan, lower_cyan)

