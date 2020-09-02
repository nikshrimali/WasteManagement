#!/usr/bin/env python
# coding: utf-8

# In[1]:


from starter import *
#Generating the training function

def gen_data(data):
    dict = {'plastic' : 1, 'glass' : 2, 'cardboard': 3, 'metal': 4, 'paper' : 5,'trash': 6}
    img = [] #List for image
    label = [] #List for labels
    
    for i in range(len(data)):     

        duplicated_imgs = preprocessing(data['filepath'].iloc[i])
        #print(duplicated_imgs.shape)
        
#         for _ in duplicated_imgs:
        img.append(duplicated_imgs)
        label.append(data['catagory'].iloc[i])
    

    valid_image = np.asarray(img)
    valid_labels = np.asarray(label)
    return valid_image, valid_labels


# In[2]:


x_train, y_train = gen_data(train_data)
x_val, y_val = gen_data(val_data)
x_test, y_test = gen_data(test_data)


# In[3]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
def onehotop(y):
    print('shape of y_input ', y.shape)
    b = np.reshape(y, (-1,1))
    print('shape of y_converted ', y.shape)
    b = enc.fit_transform(b)
    return b


# In[4]:


y_train = onehotop(y_train)
y_val = onehotop(y_val)
y_test = onehotop(y_test)


# In[ ]:




batch_size = 2
epochs = 4
model.fit(x_train, y_train,epochs=epochs, verbose = 1, callbacks= [es,chkpt], validation_data=[x_val, y_val])


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

