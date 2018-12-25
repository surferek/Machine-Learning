# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:47:40 2018

@author: Piotr
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Import digits dataset
digits_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()


# Class names storing point
class_names = ['0','1', '2', '3', '4', '5', 
               '6', '7', '8', '9']



# Inspecting image
def inspect_plt(data_image,number):
    plt.figure()
    plt.imshow(data_image[number])
    plt.colorbar()
    plt.grid(False)
    
inspect_plt(train_images,3)


# Remember to preprocess training and testing set in the same ways!!!
train_images = train_images / 255.0
test_images = test_images / 255.0


"""
    Display the first n images from the training set
     and display the class name below each image
"""
def img_label_plt(images,labels,class_names,n):
    plt.figure(figsize=(10,10))
    for i in range(n):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

img_label_plt(train_images,train_labels,class_names,4)



# Building the model
"""
    Flattening shape of out input images to 28x28 pixels into 1-d array    
"""
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    # Compiling the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Training the model
"""
    Uncomment if you run your model for the first time and you want to save it
"""    

"""
model = build_model()
model.fit(train_images, train_labels, epochs=9, workers=5, use_multiprocessing=True)
model.save('PROJEKT/model/digit_recognizer_model.h5')
"""

####################                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


"""

    Skip this part if you already trained your model or already saved one
    
"""
#    Load model from model directory including weights and optimizer.
model = keras.models.load_model('PROJEKT/model/digit_recognizer_model.h5')
model.summary()



#    Evaluate the model
"""
   Check how the model performs on the test dataset:
"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

##########################                     ################################
# Visualisation functions
def digit_plot(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
##########################                     ################################



#    Begin predictions
"""
    A prediction is an array of 10 numbers. 
    These describe the "confidence" of the model that 
    the image corresponds to each of the 10 different shapes of digits 0-9. 
"""

predictions = model.predict(test_images) # Our predictions for test_images

sample_pred=39  # n-th item from test_images
predictions[sample_pred] # Prediction array of test_images
np.argmax(predictions[sample_pred]) # Our predicted number

inspect_plt(test_images,sample_pred) # Visual check for labeled digit
test_labels[sample_pred] # Non-visual outpud of labeled digit

# Visualized proof of our prediction
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
digit_plot(sample_pred, predictions, test_labels, test_images)


# Plot for the first n test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 6
num_cols = 2

num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, i+1)
  digit_plot(i, predictions, test_labels, test_images)



####################        Custom predictions       ##########################

"""
Just open paint and create black square. 
Then draw with pencil or brush digit from 0-9 and save it.

"""

# Required libraries
from scipy import misc
import matplotlib.image as mpimg
import PIL
from PIL import Image
        

# RGB to gray scale converter
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def custom_pred(img, model):
    
    # Reshape input image to 28x28 array
    image = Image.open(img)
    hpercent = (28 / float(image.size[1]))
    wsize = int((float(image.size[0]) * float(hpercent)))
    drawing = image.resize((wsize, 28), PIL.Image.ANTIALIAS)
    
    # Convert image object into numpy array
    pix = np.array(drawing.getdata()).reshape(drawing.size[0],
                  drawing.size[1], 4)
    
    # Switch RGB scale into gray scale
    gray = rgb2gray(pix)    
    
    # Show drawed digit
    plt.imshow(gray, cmap = plt.get_cmap('gray'))
    plt.show()
    
    # Reshape array
    gray = gray.reshape(1,28,28).astype('float64')
    
    # Make prediction
    predictions = model.predict(gray)    
    cnt=0
    for i in range(10):
        if predictions[0][i] == 1:
            print("I predict that you draw: ",cnt)
        cnt+=1
        
        
        
        
custom_pred("PROJEKT/number.jpg", model)