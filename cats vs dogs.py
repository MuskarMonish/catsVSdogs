import numpy as np
import cv2 
import os
import random
import matplotlib.pyplot as plt
import pickle




DIRECTORY=r'D:\downloads\dogscats\train'
CATEGORY=['cats','dogs']




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





X = []
y = []





for features, label in data:
    X.append(features)
    y.append(label)





X = np.array(X)
y = np.array(y)





X





y





pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))





X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))




X = X/255




X = X.shape





from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten




model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))





model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])





model.fit(X, y, epochs=5, validation_split=0.1)





X.shape





import time

time.time()





import cv2
import keras





CATEGORIES = ['Cat', 'Dog']

path=r'D:\downloads\dogscats\train\dogs\dog.0.jpg'
img = cv2.imread(path)
new_arr = cv2.resize(img, (100, 100))
new_arr = np.array(new_arr)
new_arr = new_arr.reshape(-1, 100, 100, 3)





prediction = model.predict(new_arr)
print(CATEGORIES[prediction.argmax()])
