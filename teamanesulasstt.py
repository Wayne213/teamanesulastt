#!/usr/bin/env python
# coding: utf-8

# In[1]:


# R178512E ANESU CHITSIKU
# R178495N WAYNE N MAISENI
# Assignment [To be done in Pairs]


# In[3]:


#importing that l will use libraries
import math  
import cv2
import numpy as np  
import pandas as pd
$ ipython python/my_test_imagenet.py
from IPython import get_ipython
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing import image   
from keras.utils import np_utils
from skimage.transform import resize  


# In[4]:


# Uploading video, splitting it into frames so that i can train my model
count = 0
cap = cv2.VideoCapture('C:/Users/Wayne/Desktop/qwqddsfv.mp4')  
frameRate = cap.get(5)
x=1
while(cap.isOpened()):
    frameId = cap.get(1)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="vidframe%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Video uploaded and extracted images successfully!")


# In[5]:


# sample image extracted
image = plt.imread('vidframe35.jpg') 
plt.imshow(image)


# In[6]:


# my sample dataset
data = pd.read_csv('C:/Users/Wayne/Desktop/data.csv')
data.head(10)


# In[7]:


# store my images in an array
myarray = [ ]     
for img_name in data.Image_NAME:
    img = plt.imread('' + img_name)
    myarray.append(img) 
myarray = np.array(myarray)    


# In[8]:


p = data.Class
dummy_y = np_utils.to_categorical(p) 


# In[9]:


# reshaping my images
image = []
for i in range(0,myarray.shape[0]):
    a = resize(myarray[i], preserve_range=True, output_shape=(224,224)).astype(int)  
    image.append(a)
myarray = np.array(image)


# In[10]:


#image preprocessing
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
tf.keras.applications.resnet.preprocess_input(
    myarray, data_format=None
)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(myarray, dummy_y, test_size=0.3, random_state=42)   


# In[12]:


from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout


# In[13]:


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 


# In[14]:


X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape


# In[15]:


# convert images to 1-D
X_train = X_train.reshape(208, 7*7*512)
X_valid = X_valid.reshape(90, 7*7*512)


# In[16]:


# centering image data
train = X_train/X_train.max()      
X_valid = X_valid/X_train.max()


# In[17]:


# Developing the model
model = Sequential()
model.add(InputLayer((7*7*512,)))  
model.add(Dense(units=1024, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))  


# In[18]:


model.summary()


# In[19]:


# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[20]:


# Training the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))


# In[23]:


# Upoad user video

count = 0
cap = cv2.VideoCapture('C:/Users/Wayne/Desktop/index.mp4')
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="test%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("User uploaded video successfully and converted into frames!")


# In[24]:


test = pd.read_csv('C:/Users/Wayne/Desktop/test.csv')


# In[25]:


test_image = []
for img_name in test.Image_NAME:
    img = plt.imread('' + img_name)
    test_image.append(img)
test_img = np.array(test_image)


# In[26]:


test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)


# In[27]:


#image preprocessing
from keras.applications.vgg16 import preprocess_input
test_image = tf.keras.applications.resnet.preprocess_input(
    test_image, data_format=None
)


# In[28]:


# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)

# converting the images to 1-D form
test_image = test_image.reshape(186, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()


# In[29]:


predictions = np.argmax(model.predict(test_image), axis=-1)


# In[41]:


none = predictions[predictions==0].shape[0]
jerry = predictions[predictions==1].shape[0]
tom = predictions[predictions==2].shape[0]

query_search = input("Type object to search:")
if query_search.lower() == "tom":
  print("Object found in", tom, "frames")
elif query_search.lower() == "jerry":
  print("Object found in", jerry, "frames")
else:
  print("Object not found")

