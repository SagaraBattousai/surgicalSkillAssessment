import cv2
import numpy as np
import os

patchSizesStraight=[[10, 10],[20,20],[10,20],[20,10]]

patchSizesDiagonal=[[16,16],[22,22],[16,16],[22,22]]

diagonalMask=[None, None, None, None]
diagonalMaskColour=[None, None, None, None]

patchJump=10

patch2DJump=8
patch3DJump=6

diagOverlay=4


def getDiagonalMask(axis, colour=False):
    
    global diagonalMaskColour

    global diagonalMask
    
    if colour:
        if diagonalMaskColour[axis] is not None:
            return diagonalMaskColour[axis]


    else:
        if diagonalMask[axis] is not None:
            return diagonalMask[axis]

    zero = (0,0,0) if colour else 0

    one = (1,1,1) if colour else 1


    if axis ==0:
        top = [one]*10
        top.extend([zero]*6)
        top = [top] * 6

        mid = [one]*16
        mid = [mid] * 4

        bottom = [zero]*6
        bottom.extend([one]*10)
        bottom = [bottom] * 6

        mask = []
        mask.extend(top)
        mask.extend(mid)
        mask.extend(bottom)

        if colour:
            mask = np.array(mask).reshape(16,16,3)

            diagonalMaskColour[axis] = mask

            return mask

        else:
            mask = np.array(mask).reshape(16,16,1)

            diagonalMask[axis] = mask

            return mask

    if axis ==1:
        top1 = [one]*10
        top1.extend([zero]*12)
        top1 = [top1] * 6
        
        top2 = [one]*16
        top2.extend([zero]*6)
        top2 = [top2] * 4

        mid = [zero]*6
        mid.extend([one]*10)
        mid.extend([zero]*6)
        mid = [mid] * 2

        bottom1 = [zero]*6
        bottom1.extend([one]*16)
        bottom1 = [bottom1] * 4
        
        bottom2 = [zero]*12
        bottom2.extend([one]*10)
        bottom2 = [bottom2] * 6

        mask = []
        mask.extend(top1)
        mask.extend(top2)
        mask.extend(mid)
        mask.extend(bottom1)
        mask.extend(bottom2)
        
        if colour:
            mask = np.array(mask).reshape(22,22,3)

            diagonalMaskColour[axis] = mask

            return mask

        else:
            mask = np.array(mask).reshape(22,22,1)

            diagonalMask[axis] = mask

            return mask

    if axis ==2:
        top = [zero]*6
        top.extend([one]*10)
        top = [top] * 6

        mid = [one]*16
        mid = [mid] * 4

        bottom = [one]*10
        bottom.extend([zero]*6)
        bottom = [bottom] * 6

        mask = []
        mask.extend(top)
        mask.extend(mid)
        mask.extend(bottom)

        if colour:
            mask = np.array(mask).reshape(16,16,3)

            diagonalMaskColour[axis] = mask

            return mask

        else:
            mask = np.array(mask).reshape(16,16,1)

            diagonalMask[axis] = mask

            return mask

    if axis ==3:

        top1 = [zero]*12
        top1.extend([one]*10)
        top1 = [top1] * 6

        top2 = [zero]*6
        top2.extend([one]*16)
        top2 = [top2] * 4
        
        mid = [zero]*6
        mid.extend([one]*10)
        mid.extend([zero]*6)
        mid = [mid] * 2

        bottom1 = [one]*16
        bottom1.extend([zero]*6)
        bottom1 = [bottom1] * 4

        bottom2 = [one]*10
        bottom2.extend([zero]*12)
        bottom2 = [bottom2] * 6
        
        mask = []
        mask.extend(top1)
        mask.extend(top2)
        mask.extend(mid)
        mask.extend(bottom1)
        mask.extend(bottom2)

        if colour:
            mask = np.array(mask).reshape(22,22,3)

            diagonalMaskColour[axis] = mask

            return mask

        else:
            mask = np.array(mask).reshape(22,22,1)

            diagonalMask[axis] = mask

            return mask

def patches(image, colour=False):

    x, y, _ = image.shape

    patches1 = [[image[i:i+patchSizesStraight[0][0], j:j+patchSizesStraight[0][1]]\
            for i in range(0,x,patchJump)]for j in range(0,y,patchJump)]

    patches2 = [[image[i:i+patchSizesStraight[1][0], j:j+patchSizesStraight[1][1]]\
            for i in range(0,x-10,patchJump)]for j in range(0,y-10,patchJump)]

    patches3 = [[image[i:i+patchSizesStraight[2][0], j:j+patchSizesStraight[2][1]]\
            for i in range(0,x,patchJump)]for j in range(0,y-10,patchJump)]

    patches4 = [[image[i:i+patchSizesStraight[3][0], j:j+patchSizesStraight[3][1]]\
            for i in range(0,x-10,patchJump)]for j in range(0,y,patchJump)]

    patches5 = [[image[i:i+patchSizesDiagonal[0][0], j:j+patchSizesDiagonal[0][1]]\
            for i in range(0,x-8,patch2DJump)]for j in range(0,y-8,patch2DJump)]

    patches6 = [[image[i:i+patchSizesDiagonal[1][0], j:j+patchSizesDiagonal[1][1]]\
            for i in range(0,x-16,patch3DJump)]for j in range(0,y-16,patch3DJump)]

    patches7 = [[image[i:i+patchSizesDiagonal[2][0], j:j+patchSizesDiagonal[2][1]]\
            for i in range(0,x-8,patch2DJump)]for j in range(0,y-8,patch2DJump)]

    patches8 = [[image[i:i+patchSizesDiagonal[3][0], j:j+patchSizesDiagonal[3][1]]\
            for i in range(0,x-16,patch3DJump)]for j in range(0,y-16,patch3DJump)]

    p1 =np.array(patches1).reshape(4,4,10,10,3).astype(np.uint8)
    p2 =np.array(patches2).reshape(3,3,20,20,3).astype(np.uint8)
    p3 =np.array(patches3).reshape(3,4,10,20,3).astype(np.uint8)
    p4 =np.array(patches4).reshape(4,3,20,10,3).astype(np.uint8)

    
            
    if colour:
        p5 = (np.array(patches5) * getDiagonalMask(0, colour=True).astype(np.uint8)).reshape(4,4,16,16,3)
        p6 = (np.array(patches6) * getDiagonalMask(1, colour=True).astype(np.uint8)).reshape(4,4,22,22,3)
        p7 = (np.array(patches7) * getDiagonalMask(2, colour=True).astype(np.uint8)).reshape(4,4,16,16,3)
        p8 = (np.array(patches8) * getDiagonalMask(3, colour=True).astype(np.uint8)).reshape(4,4,22,22,3)

    else:
        p5 = (np.array(patches5) * getDiagonalMask(0).astype(np.uint8)).reshape(4,4,16,16,3)
        p6 = (np.array(patches6) * getDiagonalMask(1).astype(np.uint8)).reshape(4,4,22,22,3)
        p7 = (np.array(patches7) * getDiagonalMask(2).astype(np.uint8)).reshape(4,4,16,16,3)
        p8 = (np.array(patches8) * getDiagonalMask(3).astype(np.uint8)).reshape(4,4,22,22,3)

    return [p1,p2,p3,p4,p5,p6,p7,p8]

def imageAction(key, imageDir, image, counter, label, imageFormat="frame_{}.jpg", lname="{}\n"):

    while key != ord('q') and key != ord('t') and key != ord('f') and key != ord('b'):
        key = cv2.waitKey(0) & 0xFF
    
    cv2.imwrite(imageDir + os.sep + imageFormat.format(counter), image)

    if key == ord('q'):
        label.write(lname.format(0))
        label.flush()
        print('Added {} as Empty'.format(counter))

    elif key == ord('t'):
        label.write(lname.format(1))
        label.flush()
        print('Added {} as Thumb'.format(counter))

    elif key == ord('f'):
        label.write(lname.format(2))
        label.flush()
        print('Added {} as Finger'.format(counter))

    elif key == ord('b'):
        label.write(lname.format(1))
        label.write(lname.format(2))
        label.flush()
        print('Added {} as Thumb'.format(counter))
        
        counter += 1
        cv2.imwrite(imageDir + os.sep + imageFormat.format(counter), image)
        print('Added {} as Finger'.format(counter))

    counter += 1
    
    cv2.destroyAllWindows()
    return counter




def handHelper(image, x, y, nameCounter, imgDir, label, lname, colour=True):

    imagePatches = patches(image, colour=colour)

    level = 0

    for u, j in enumerate(range(0,x,patchJump)):
        for v, i in enumerate(range(0,y,patchJump)):
            img2 = cv2.rectangle(image.copy(), (i, j),         
                    (i + patchSizesStraight[0][0], 
                        j + patchSizesStraight[0][1]), (255, 255, 0))
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF

            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    level = 1

    for u, j in enumerate(range(0,x-10,patchJump)):
        for v, i in enumerate(range(0,y-10,patchJump)):

            img2 = cv2.rectangle(image.copy(), (i, j),         
                    (i + patchSizesStraight[1][0], 
                        j + patchSizesStraight[1][1]), (255, 255, 0))
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF
            
            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    level = 3

    for u, j in enumerate(range(0,x,patchJump)):
        for v, i in enumerate(range(0,y-10,patchJump)):

            img2 = cv2.rectangle(image.copy(), (i, j),         
                    (i + patchSizesStraight[2][0], 
                        j + patchSizesStraight[2][1]), (255, 255, 0))
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF
            
            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    level = 2

    for u, j in enumerate(range(0,x-10,patchJump)):
        for v, i in enumerate(range(0,y,patchJump)):

            img2 = cv2.rectangle(image.copy(), (i, j),         
                    (i + patchSizesStraight[3][0], 
                        j + patchSizesStraight[3][1]), (255, 255, 0))
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF
            
            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    level = 4

    for u, j in enumerate(range(0,x-8,patch2DJump)):
        for v, i in enumerate(range(0,y-8,patch2DJump)):

            img2 = image.copy()

            img2 = cv2.line(img2, (i, j), (i + 10, j), (255, 255, 0), 1)
            
            img2 = cv2.line(img2, (i + 10, j), (i + 10, j + 6) ,(255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i + 10, j + 6), (i+16, j+6) ,(255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+16, j+6), (i+16, j+16), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+16, j+16), (i+6, j+16), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+6, j+16), (i+6, j+10), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+6, j+10), (i, j+10), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i, j+10), (i,j),(255, 255, 0), 1)
                
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF
            
            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    level = 5

    for u, j in enumerate(range(0,x-16,patch3DJump)):
        for v, i in enumerate(range(0,y-16,patch3DJump)):

            img2 = image.copy()

            img2 = cv2.line(img2, (i, j), (i + 10, j), (255, 255, 0), 1)
            
            img2 = cv2.line(img2, (i + 10, j), (i + 10, j + 6) ,(255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i + 10, j + 6), (i+16, j+6) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+16, j+6), (i+16,j+12) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+16,j+12), (i+22, j+12) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+22, j+12), (i+22, j+22) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+22, j+22), (i+12, j+22) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+12, j+22), (i+12, j+16) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+12, j+16), (i+6, j+16) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+6, j+16), (i+6, j+10), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+6, j+10), (i, j+10), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i, j+10), (i,j),(255, 255, 0), 1)
                
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF
            
            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    level = 6

    for u, j in enumerate(range(0,x-8,patch2DJump)):
        for v, i in enumerate(range(0,y-8,patch2DJump)):

            img2 = image.copy()

            img2 = cv2.line(img2, (i+6, j), (i + 16, j), (255, 255, 0), 1)
            
            img2 = cv2.line(img2, (i + 16, j), (i + 16, j + 10) ,(255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i + 16, j + 10), (i+10, j+10) ,(255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+10, j+10), (i+10, j+16), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+10, j+16), (i+0, j+16), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+0, j+16), (i+0, j+6), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+0, j+6), (i+6, j+6), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+6, j+6), (i+6,j),(255, 255, 0), 1)
                
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF
            
            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    level = 7

    for u, j in enumerate(range(0,x-16,patch3DJump)):
        for v, i in enumerate(range(0,y-16,patch3DJump)):

            img2 = image.copy()

            img2 = cv2.line(img2, (i+12, j), (i + 22, j), (255, 255, 0), 1)
            
            img2 = cv2.line(img2, (i + 22, j), (i + 22, j + 10) ,(255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i + 22, j + 10), (i+16, j+10) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+16, j+10), (i+16,j+16) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+16,j+16), (i+10, j+16) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+10, j+16), (i+10, j+22) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+10, j+22), (i, j+22) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i, j+22), (i, j+12) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i, j+12), (i+6, j+12) ,(255, 255, 0), 1)

            img2 = cv2.line(img2, (i+6, j+12), (i+6, j+6), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+6, j+6), (i+12, j+6), (255, 255, 0), 1)
                
            img2 = cv2.line(img2, (i+12, j+6), (i+12,j),(255, 255, 0), 1)
                
                
            cv2.imshow('image_{}'.format(nameCounter), img2)

            key = cv2.waitKey(0) & 0xFF
            
            nameCounter = imageAction(key, imgDir, imagePatches[level][u,v], nameCounter, label)

    return nameCounter




def HandToTip(startAt=0,baseDir='labels', imgDir="images"):
    try:
        labelFile = open('labels.txt', 'a')
        lname = "{}\n"
        counter = 1

        images = os.listdir(baseDir)
        images.sort(key=lambda name: int(name[6:-4]))

        image_number = startAt + 1
        num_images = len(images)
    
        if not os.path.exists(imgDir):
            os.mkdir(imgDir)

        for img in images[startAt:]:

            labelFile.write("#" * 80 + "\n")
            labelFile.flush()

            image = cv2.imread(baseDir + os.sep + img)

            h, w, _ = image.shape

            print('You are Currently on image {} of {}, ie {:.2%} done'.format(image_number - 1, num_images, image_number / num_images))

            image_number += 1

            counter = handHelper(image, h, w, counter, imgDir, labelFile, lname)
            
        labelFile.write("#" * 80 + "\n")
        labelFile.flush()
            

    finally:
        cv2.destroyAllWindows()
        labelFile.close()
# def patches(image, colour=False):

#     x, y, _ = image.shape


#     p1 =np.array(patches1)
#     p2 =np.array(patches2)
#     p3 =np.array(patches3)
#     p4 =np.array(patches4)

    
            
#     if colour:
#         p5 = np.array(patches5) * getDiagonalMask(0, colour=True)
#         p6 = np.array(patches6) * getDiagonalMask(1, colour=True)
#         p7 = np.array(patches7) * getDiagonalMask(2, colour=True)
#         p8 = np.array(patches8) * getDiagonalMask(3, colour=True)

#     else:
#         p5 = np.array(patches5) * getDiagonalMask(0)
#         p6 = np.array(patches6) * getDiagonalMask(1)
#         p7 = np.array(patches7) * getDiagonalMask(2)
#         p8 = np.array(patches8) * getDiagonalMask(3)

#     return [p1,p2,p3,p4,p5,p6,p7,p8]





















if __name__ == '__main__':
    pass


