#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import cv2 
import os
import random
import matplotlib.pyplot as plt
import pickle


# In[8]:


DIRECTORY=r'D:\downloads\dogscats\train'
CATEGORY=['cats','dogs']


# In[76]:


img_size=100
data=[]
for category in CATEGORY:
    folder=os.path.join(DIRECTORY,category)
    label=CATEGORY.index(category)
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        img_arr=cv2.imread(img_path)
        img_arr=cv2.resize(img_arr,(img_size,img_size))
        data.append([img_arr,label])


# In[10]:


data


# In[11]:


data[0][1]


# In[12]:


random.shuffle(data)


# In[13]:


X = []
y = []


# In[14]:


for features, label in data:
    X.append(features)
    y.append(label)


# In[15]:


X = np.array(X)
y = np.array(y)


# In[16]:


X


# In[17]:


y


# In[18]:


pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))


# In[51]:


X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))


# In[52]:


X


# In[53]:


X = X/255


# In[54]:


X


# In[23]:


X = X.shape


# In[24]:


X


# In[35]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[55]:


model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))


# In[56]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[57]:


model.fit(X, y, epochs=5, validation_split=0.1)


# In[58]:


X.shape


# In[59]:


import time

time.time()


# In[62]:


import cv2
import keras


# In[106]:


import numpy as np
CATEGORIES = ['Cat', 'Dog']

path=r'D:\downloads\dogscats\train\dogs\dog.0.jpg'
img = cv2.imread(path)
new_arr = cv2.resize(img, (100, 100))
new_arr = np.array(new_arr)
new_arr = new_arr.reshape(-1, 100, 100, 3)


# In[107]:


prediction = model.predict(new_arr)
print(CATEGORIES[prediction.argmax()])
