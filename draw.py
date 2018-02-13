import numpy as np
import cv2

img = np.zeros((512, 512, 3), np.uint8)

image_center = 512.0 / 2

tri_radius = (image_center)/3

redCvCenter = (int(image_center), int(image_center - tri_radius))
greenCvCenter = (int(image_center - tri_radius), int(image_center + tri_radius))
blueCvCenter = (int(image_center + tri_radius), int(image_center + tri_radius))

cvRadius = int(image_center / 4)

img = cv2.ellipse(img, redCvCenter, (cvRadius, cvRadius), 0, 105, 75, (0,0,255), -1)   


cv2.imwrite("draw.png", img)

