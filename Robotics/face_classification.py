from matplotlib import pyplot as plt
import numpy as np
import cv2

def face_detection(gray):
    """
    Face detection function
    \return: result is the bounding box position in (x1, y1, x2, y2)
    """
    # convert to the original
    gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    
    faces_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # sliding window detection, 1.3 is the scaling factor of the window 
    # return the region contains the faces
    faces = faces_cascade.detectMultiScale(gray,1.3,5)
    result=[]
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))

    return result


def parse_bounding_box(img_name):
    """
    Draw bounding boxes on picture
    """
    img = cv2.imread(img_name)
    faces = face_detection(img)
    if faces:
        for (x1,y1,x2,y2) in faces:
            # append face region to the original image
            cv2.rectangle(img, (x1,y1),(x2,y2),(255,255,0), thickness=8)
            
            # save the face region
            face_region = img[y1:y2,x1:x2]
    else:
        print("face not detected")

    # show picture, or you can use cv2.imshow('face', img)
    plt.imshow(img,'gray')
    plt.show()

if __name__=='__main__':
    filename = 'picture/pic001.jpg'
    parse_bounding_box(filename)