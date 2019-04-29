#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from sklearn.linear_model import LogisticRegression
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
# train_test_split_ratio = 0.7


# In[19]:


from feature_selection_ga import FeatureSelectionGA
rotationforest = rotation_forest.RotationForestClassifier(random_state=False,max_depth=4,n_estimators=10)


# In[21]:


#train_X, train_Y => diagnostic dataset 1 => 30 features genetic algorithm use
#train_X2, train_Y2 => original dataset 2 => 10 features


# In[4]:


# !pip3 install seaborn


# # Dataset 2 Read

# In[5]:


data2 = np.genfromtxt("./data/breast-cancer-wisconsin.data",delimiter=",")


# In[6]:


train_data2 = data2.T[1:].T
train_X2 = train_data2.T[0:-1].T.astype(float)
train_Y2 = train_data2.T[-1].T
for i in range(len(train_Y2)):
    if(train_Y2[i] == 2):
        train_Y2[i] = 1
    else:
        train_Y2[i] = 0
# print (train_Y2)
# print len(train_X2)
# print len(train_X2[0])
train_X2 = np.array(train_X2)
train_Y2 = np.array(train_Y2)
train_X2,train_Y2 = shuffle(train_X2,train_Y2)


# # Dataset 1 read

# In[22]:


completedata = pd.read_csv("./data/data.csv")
completedata.head()
completedata.loc[completedata.diagnosis == 'M', 'diagnosis'] = 1
completedata.loc[completedata.diagnosis == 'B', 'diagnosis'] = 0


# In[23]:


print completedata.keys()


# In[24]:


train_XGA = completedata[["radius_se","texture_mean","perimeter_mean","perimeter_se","area_se","smoothness_se","compactness_worst","concavity_mean","concavity_worst","concave points_se","concave points_worst","symmetry_mean","symmetry_worst","fractal_dimension_mean"]]
train_YGA = completedata.diagnosis
train_Y = completedata.diagnosis
train_X = completedata.drop(["id","Unnamed: 32","diagnosis"],axis=1)


# In[25]:


train_XGA.head()


# In[27]:


train_Y = np.array(train_Y)
train_X = np.array(train_X)
train_YGA = np.array(train_YGA)
train_XGA = np.array(train_XGA)
train_X,train_Y,train_XGA,train_YGA = shuffle(train_X,train_Y,train_XGA,train_YGA)


# In[10]:


# train_X.iloc()


# In[11]:


# print (train_X.shape)
# # print (completedata.shape)
# print (train_X[0:3,:])


# In[12]:


# train_X.head()


# In[13]:


train_Y.shape


# In[14]:


# mask = np.zeros_like(train_X.corr(), dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# plt.figure(figsize=(16,8))
# sb.heatmap(np.abs(train_X.corr()),cmap='YlGn',mask=mask, annot=True)
# plt.show()


# In[15]:


# ax = sb.countplot(train_Y,label="Count")       # M = 212, B = 357
# B, M = train_Y.value_counts()
# print('Number of Benign: ',B)
# print('Number of Malignant : ',M)


# # Dataset read complete

# In[16]:


import rotation_forest
def reports(classifier,train_data,train_labels,train_test_split_ratio=.2):
    scores = []
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels,shuffle=False, test_size=train_test_split_ratio)
    kf = KFold(n_splits=5,shuffle=False)
    kf.get_n_splits(X_train)
    print(kf)
    scores = []
    for train_index, test_index in kf.split(X_train):
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_traini, X_testi = X_train[train_index], X_train[test_index]
        y_traini, y_testi = y_train[train_index], y_train[test_index]
        classifier.fit(X_traini,y_traini)
        predicted = classifier.predict(X_testi)
        scores.append(accuracy_score(predicted,y_testi))
    scores = np.array(scores)
    print ("Average Accuracy K Fold: ",scores.mean())
    
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    print ("Test Data Results:")
    print ("Test Accuracy: ",accuracy_score(predicted,y_test))
    X = classification_report(y_test,predicted)
    print (X)
#     print ("MCC: ",mcc(test_y,predicted))
#     print ("")


# In[ ]:





# In[17]:


from sklearn.decomposition import PCA
pca = PCA(n_components=25)


# In[28]:


reports(rotationforest,train_XGA,train_YGA,0.2)


# In[55]:


train_Y[0:5]


# In[175]:


ga = FeatureSelectionGA(rotationforest,train_X,train_Y,cv_split=10)
pop = ga.generate(100)


# In[ ]:





# In[1]:


# print (pop)


# In[112]:


# print data2[0]


# In[125]:


# train_data2 = data2.T[1:].T
# train_X2 = []
# train_Y2 = []

# for i in range(len(train_data2)):
#     if(np.nan not in train_data2[i]):
#         train_X2.append(train_data2[i][0:-1])
#         if(train_data2[i][-1] == 2):
#             train_Y2.append(1)
#         else:
#             train_Y2.append(0)

# # train_Y2 = train_data2.T[-1].T
# # for i in range(len(train_Y2)):
# #     if(train_Y2[i] == 2):
# #         train_Y2[i] = 1
# #     else:
# #         train_Y2[i] = 0
# print (train_Y2)
# print len(train_X2)
# # print len(train_X2[0])


# In[164]:


ga = FeatureSelectionGA(rotationforest,train_X2,train_Y2,cv_split=10)
pop = ga.generate(50,mutxpb=0.4,ngen=6,cxpb=0.6)


# In[129]:


rotationforest.fit(train_X2,train_Y2)


# In[130]:


y = rotationforest.predict(train_X2)


# In[132]:


accuracy_score(y_pred=y,y_true=train_Y2)


# In[168]:


reports(rotationforest,train_X,train_Y,0.2)


# In[107]:


rotationforest = rotation_forest.RotationForestClassifier(random_state=False,max_depth=4,n_estimators=10)
reports(rotationforest,train_X2,train_Y2,0.2)


# In[23]:


print len(train_X2)


# In[121]:


logistic = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
reports(logistic,train_X2,train_Y2,0.2)


# In[144]:


from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(random_state=0,max_depth=4)
reports(decisiontree,train_X2,train_Y2,0.2)


# In[136]:


ann = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
reports(ann,train_X2,train_Y2,0.2)


# In[136]:


rbfn
reports(rbfn,train_X2,train_Y2,0.2)


# In[129]:


randomforest = RandomForestClassifier(n_estimators=10, max_depth=4,random_state=0)
reports(randomforest,train_X2,train_Y2,0.2)


# In[150]:


gausian = GaussianNB()
reports(gausian,train_X2,train_Y2,0.2)


# In[143]:


svm = SVC(gamma='auto',kernel='poly')
reports(svm,train_X2,train_Y2,0.2)


# In[ ]:




