# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessing import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreProcessor
from datasets.datasetLoader import SimpleDatasetLoader
from nn.conv.shallowNet import ShallowNet
from tensorflow import keras
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


#construct the argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="Path to dataset")
args=vars(ap.parse_args())

#grab the list of images 
print("[INFO] loading images...")
imagePaths=list(paths.list_images(args['dataset']))


#initialize the image preprocessors
sp=SimplePreProcessor(32,32)
iap=ImageToArrayPreprocessor()

#load the dataset from the disk then scale the raw pixel 
# intensities to the range [0,1]
sdl=SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.astype("float")/255.0

#partition the data into training and testing
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=42)


#convert the labels from integers to vectors

trainY=LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)


#initialize the optimizer and the model
print("[INFO] compiling the model.....")
opt=SGD(learning_rate=0.005)
model=ShallowNet.build(width=32,height=32,depth=3,classes=3)

model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])


#train the network
print("[INFO] training network...")

'''
    Here are the possible values for verbose:
        0: Silent mode. No output will be displayed during training.
        1: Progress bar mode. A progress bar will be displayed 
           showing the progress of each epoch.
        2: One line per epoch. A detailed message will be printed 
           at the end of each epoch.
'''
H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=100,verbose=1)


# evaluate the network
print("[INFO] evaluating the network....")
predictions=model.predict(testX,batch_size=32)
'''
    *testY.argmax(axis=1):
    -testY is the true labels of the test dataset in one-hot encoded 
    format.
    -argmax(axis=1) converts these one-hot encoded vectors back to 
    their original class labels (integers). It finds the index of the 
    maximum value along axis 1 (i.e., the class with the highest 
    probability).
    -For example, if testY is [[0, 0, 1], [1, 0, 0], [0, 1, 0]], 
    testY.argmax(axis=1) would be [2, 0, 1].
    
    *predictions.argmax(axis=1):
    -predictions are the predicted probabilities for each class,
      produced by the model.
    -argmax(axis=1) converts these probability vectors to predicted 
    class labels (integers) by selecting the class with the highest 
    predicted probability.
    -For example, if predictions is [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], 
    [0.2, 0.5, 0.3]], predictions.argmax(axis=1) would be [2, 0, 1].
    target_names=['cat', 'dog', 'panda']:

    -This argument specifies the names of the classes in the order 
    corresponding to their integer labels.
    -If the class labels are [0, 1, 2], target_names=['cat', 'dog', 
    'panda'] means 0 corresponds to "cat", 1 corresponds to "dog", 
    and 2 corresponds to "panda".
'''
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=['cat','dog','panda']))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()