import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from keras.models import load_model
import argparse
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("-mp","--model_path",required=True,help="Path to model")
ap.add_argument("-i","--image",type=str,required=True,help="Path to the input image")
args=vars(ap.parse_args())
# load the model
loaded_model=load_model(args["model_path"])

labelNames=["airplane","automobile","bird","cat","deer"
            ,"dog","frog","horse","ship","truck"]
#load and preprocess the image
img=cv2.imread(args["image"])
img=cv2.resize(img,(32,32))
img=img_to_array(img)
img=img.astype("float32")/255.0
print(img.shape)
'''
    The line image = np.expand_dims(image, axis=0) is used 
    to add an extra dimension to the image array. 
    This is necessary because the model expects a batch of 
    images as input, even if you're only predicting on a single image.
'''
img=np.expand_dims(img,axis=0)
print(f"After: {img.shape}")
prediction_img=loaded_model.predict(img)
prediction_Class=prediction_img.argmax(axis=1)[0]
predicted_label=labelNames[prediction_Class]
print(f"the predicted class is:{predicted_label}")

output_img=cv2.imread("dog.jpeg")
cv2.putText(output_img,predicted_label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
cv2.imshow("Prediction",output_img)
cv2.imwrite("Output_img.jpg",output_img)
cv2.waitKey()
cv2.destroyAllWindows()