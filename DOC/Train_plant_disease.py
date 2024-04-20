#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')


# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Data Preprocessing

# ### Training Image Preprocessing

# In[3]:


training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
#     pad_to_aspect_ratio=False,
#     data_format=None,
#     verbose=True,
)


# ### Validation Image Preprocessing

# In[4]:


validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
#     pad_to_aspect_ratio=False,
#     data_format=None,
#     verbose=True,
)


# In[5]:


training_set


# In[6]:


for x,y in training_set:
    print(x)
    print(y)
    break


# In[7]:


for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


# ### To avoid overshooting
# 1. Choose small learning rate default 0.001 , we are taking 0.0001
# 2. There may be chance of underfitting, so increase number of neuron
# 3. Add more Convolution layer to extract more important feature from images, there may be possibility that model unable to capture relevant feature or model is confusing due to lack of feature so feed with more feature

# ## Building Model

# In[8]:


from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten, Dropout
from tensorflow.keras.models import Sequential


# In[9]:


model=Sequential()


# ## Building Convolutional Layer

# In[10]:


model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[11]:


model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[12]:


model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[13]:


model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[14]:


model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[15]:


model.add(Dropout(0.25)) #To avoid overfitting


# In[16]:


model.add(Flatten())


# In[17]:


model.add(Dense(units=1500,activation='relu'))


# In[18]:


model.add(Dropout(0.4))


# In[19]:


## Output Layer
model.add(Dense(units=38,activation='softmax'))


# ## Compiling Model

# In[20]:


model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[21]:


model.summary()


# ## Model Training

# In[22]:


training_history=model.fit(x=training_set,validation_data=validation_set,epochs=10)


# ## Model Evaluation

# In[23]:


#Model Evaluation on Training set
train_loss, train_acc = model.evaluate(training_set)
#Training set Accuracy
print('Training accuracy:', train_acc)


# In[24]:


print(train_loss,train_acc)


# In[33]:


#Validation set Accuracy
val_loss, val_acc = model.evaluate(validation_set)
print('Validation accuracy:', val_acc)


# ### Saving Model

# In[38]:


model.save("trained_model.h5")
model.save('trained_plant_disease_model.keras') #it is used to compress file size


# In[39]:


training_history.history #Return Dictionary of history


# In[40]:


#Recording History in json
import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)


# In[41]:


print(training_history.history.keys())


# ### Accuracy Visualization

# In[43]:


epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy Result')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()


# ### Some other metrics for model evaluation

# In[44]:


class_name = validation_set.class_names
class_name


# In[45]:


test_set = tf.keras.utils.image_dataset_from_directory(
    'valid', #not test directory
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False, #not true coz it will be passed sequentially 
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


# In[46]:


y_pred = model.predict(test_set)
y_pred, y_pred.shape


# In[49]:


predicted_categories = tf.argmax(y_pred, axis=1) #argamx() extract maximum value from above prob arr and return its index
#axis=1 means scan vertically


# In[50]:


predicted_categories


# In[51]:


true_categories = tf.concat([y for x, y in test_set], axis=0)


# In[52]:


true_categories


# In[53]:


Y_true = tf.argmax(true_categories, axis=1)


# In[54]:


Y_true


# In[56]:


get_ipython().system('pip install scikit-learn')
from sklearn.metrics import confusion_matrix,classification_report


# In[57]:


# Precision Recall Fscore
print(classification_report(Y_true,predicted_categories,target_names=class_name))


# In[60]:


cm = confusion_matrix(Y_true,predicted_categories)
#cm.shape
cm


# ### Confusion Matrix Visualization

# In[62]:


#sns.heatmap(cm)


# In[66]:


plt.figure(figsize=(40, 40))
sns.heatmap(cm,annot=True,annot_kws={"size": 20})

plt.xlabel('Predicted Class',fontsize = 30)
plt.ylabel('Actual Class',fontsize = 30)
plt.title('Plant Disease Prediction Confusion Matrix',fontsize = 40)
plt.show()


# In[ ]:





# In[ ]:




