from sklearn.linear_model import LogisticRegression
import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_PATH = "picture"

def read_images_one(index, random_pad=True, random_distort=False):
    # read in images
    image_name = DATA_PATH + '/pic00' + str(index+1) + '.jpg'
    image = cv2.imread(image_name, 1)
    
    # convert to gray and reshape
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512,512))
    
    return image
    
    
def read_images():
    train = []
    label = []

    for i in range(5):
        image_name = DATA_PATH + '/pic00' + str(i+1) + '.jpg'
        image = cv2.imread(image_name, 0)
        image = cv2.resize(image, (100, 100))
        train.append(np.array(image).reshape(-1))
        label.append(i)
        
    train = np.vstack(train)
    label = np.vstack(label)

    return train, label


def regression(train, label):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial')
    
    # perform training 
    clf.fit(train, label)
    
    # perform testing
    print("training score", clf.score(train, label))
    
    return clf


def test(clf, picture):
    # read in a picture, padding and convert to gray scale
    image = cv2.imread('picture/pic00' + str(1) + '.jpg', 1)
    image = cv2.copyMakeBorder(image,500,200,300,100,cv2.BORDER_CONSTANT,value=[180,180,180])
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
    result = clf.predict(picture.reshape(1,-1))
    prob_all = clf.predict_proba(picture.reshape(1,-1))[0]
    
    # simple outlier detection
    confidence = prob_all[result]/sum(prob_all)
    if(confidence < 0.99):
        result = -1
#         print("No faces have been detected", prob_all)
    else:
        pass
#         print("predicted class {} with confidence {}".format(result, confidence))
        
    return result


if __name__=='__main__':
    # read in image
    train, label = read_images()
    
    # perform training
    clf = regression(train, label)
    
    # perform testing
    image = read_images_one(3, random_pad=True)
    
    # object detection
    stepSize = 64
    windowSize = (100,100)
    major_class = []
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            slide_win = image[y:y + windowSize[1], x:x + windowSize[0]]   
            if slide_win.shape == (100,100):
                result = test(clf, np.array(slide_win).reshape(1,10000))
            
            if(result!=-1):
                major_class.append(result[0])
                
    # return majority class
    if len(major_class) != 0:
        major = np.argmax(np.bincount(np.array(major_class, dtype=np.int32)))
        print("class {} has been detected".format(major))
    