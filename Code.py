#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import copy
# import cv2
import os
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import seaborn as seab
from numpy import histogram
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import classification_report
# from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
import scipy
from scipy.misc import imread
# from skimage.color import rgb2lab
# from skimage.color import rgb2gray
# from skimage.measure import regionprops
import pickle
import random
import seaborn as sb
# from skimage.feature import hog,local_binary_pattern
from sklearn.model_selection import train_test_split
# from skimage import data, exposure
train_test_split_ratio = 0.7


# In[2]:


# !pip3 install seaborn


# In[42]:


completedata = pd.read_csv("./data/data.csv")
completedata.head()
completedata.loc[completedata.diagnosis == 'M', 'diagnosis'] = 1
completedata.loc[completedata.diagnosis == 'B', 'diagnosis'] = 0


# In[43]:


train_Y = completedata.diagnosis
train_X = completedata.drop(["id","Unnamed: 32","diagnosis"],axis=1)


# In[44]:


train_Y = np.array(train_Y)
train_X = np.array(train_X)


# In[45]:


# train_X.iloc()


# In[46]:


print (train_X.shape)
# print (completedata.shape)
print (train_X[0:3,:])


# In[47]:


# train_X.head()


# In[48]:


train_Y.shape


# In[49]:


# mask = np.zeros_like(train_X.corr(), dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# plt.figure(figsize=(16,8))
# sb.heatmap(np.abs(train_X.corr()),cmap='YlGn',mask=mask, annot=True)
# plt.show()


# In[50]:


# ax = sb.countplot(train_Y,label="Count")       # M = 212, B = 357
# B, M = train_Y.value_counts()
# print('Number of Benign: ',B)
# print('Number of Malignant : ',M)


# In[51]:


import rotation_forest
def reports(classifier,train_data,train_labels,train_test_split_ratio=.3):
    kf = KFold(n_splits=5)
    kf.get_n_splits(train_data)
    print(kf)
    scores = []
    for train_index, test_index in kf.split(train_data):
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        classifier.fit(X_train,y_train)
        predicted = classifier.predict(X_test)
        scores.append(accuracy_score(predicted,y_test))
    scores = np.array(scores)
    print ("Average Accuracy K Fold: ",scores.mean())
    train_data_len = len(train_data)
    chunksize = int(train_data_len*train_test_split_ratio)
    
    train_x = train_data[0:chunksize]
    train_y = train_labels[0:chunksize]

    test_x = train_data[chunksize:train_data_len]
    test_y = train_labels[chunksize:train_data_len]
    
    classifier.fit(train_x,train_y)
    predicted = classifier.predict(test_x)
    print ("Test Data Results:")
    print ("Test Accuracy: ",accuracy_score(predicted,test_y))
    X = classification_report(test_y,predicted)
    print (X)
    print ("MCC: ",mcc(test_y,predicted))
    print ("")


# In[52]:


rotationforest = rotation_forest.RotationForestClassifier()


# In[53]:


from sklearn.decomposition import PCA
pca = PCA(n_components=25)


# In[54]:


reports(rotationforest,pca.fit_transform(np.array(train_X)),np.array(train_Y),0.3)


# In[33]:


# np.array(train_Y)


# In[34]:


# train_X[0:5]


# In[55]:


train_Y[0:5]


# In[56]:


from feature_selection_ga import FeatureSelectionGA


# In[57]:


ga = FeatureSelectionGA(rotationforest,train_X,train_Y)


# In[61]:


pop = ga.generate(150)


# In[ ]:


ga.eva

