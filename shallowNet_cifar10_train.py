from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.shallowNet import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import cv2

#load the training and testing data, then scale it into the range [0,1]
print("[INFO] loading CIFAR-10...")
((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX=trainX.astype("float32")/255.0
testX=testX.astype("float32")/255.0

#convert the labels from integers to vectors
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.fit_transform(testY)

#initialize the label names for the CIFAR-10 dataset
labelNames=["airplane","automobile","bird","cat","deer"
            ,"dog","frog","horse","ship","truck"]

#initialize the optimizer and model
print("[INFO] compiling model.....")
opt=SGD(learning_rate=0.01)
model=ShallowNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])


# train the network
print("[INFO] training network...")
H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=40,verbose=1)


#evaluate the network
print("[INFO] evaluating the network....")
predictions=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))

# Save the model to disk
model_path = "shallowNet_cifar10.h5"
model.save(model_path)
print(f"[INFO] model saved to {model_path}")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
