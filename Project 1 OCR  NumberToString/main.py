import cv2
import numpy as np
from matplotlib import pyplot as plt


from keras.models import model_from_json


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")


w=640
h=480

cap=cv2.VideoCapture(0)
cap.set(3,w)
cap.set(4,h)

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    img = img/255
    return img

img1=cv2.imread("a.jpeg",1)
img=np.asarray(img1)
img=cv2.resize(img,(32,32))
img=preProcessing(img)
cv2.imshow("s",img)
img=img.reshape(1,32,32,1)
pre=loaded_model.predict(img)
print(pre)
    


# load weights into new model
# Write the file name of the weights




print('Prediction Score:\n',pre[0])



thresholded = (pre>0.5)*1
print('\nThresholded Score:\n',thresholded[0])

print('\nPredicted Digit:\n',np.where(thresholded == 1)[1][0])
