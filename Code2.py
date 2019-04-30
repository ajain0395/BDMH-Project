
import numpy as np
import copy

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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_curve,precision_recall_curve,accuracy_score,roc_auc_score
import seaborn as seab
import matplotlib.pyplot as plt
import matplotlib.patches as pat


warnings.filterwarnings("ignore")

import pickle
import random
import seaborn as sb
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split




import rotation_forest
from feature_selection_ga import FeatureSelectionGA
from sklearn.tree import DecisionTreeClassifier


####################CLASSIFIERS##########################
rotationforest = rotation_forest.RotationForestClassifier(random_state=False,max_depth=4,n_estimators=10)
decisiontree = DecisionTreeClassifier(random_state=0,max_depth=4)
logistic = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
ann = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
randomforest = RandomForestClassifier(n_estimators=10, max_depth=4,random_state=0)
gausian = GaussianNB()
sv=svm.SVC(gamma='scale')
vote= VotingClassifier(estimators=[('lr', logistic),('rf',rotationforest),('dt',decisiontree)], voting='hard')
#svm = SVC(gamma='auto',kernel='poly')
knn = KNeighborsClassifier(weights='uniform',n_neighbors=48,p=2,leaf_size=30,metric='minkowski',n_jobs=1,algorithm='auto',metric_params=None)





def heatmap(confusionmat,title="Confusion Matrix",title2="",index=0):
    plt.figure()
    ax = plt.axes()
    seab.heatmap(confusionmat,linewidths=0.4,linecolor='white',annot=True,fmt='g') 
    ax.set_title(title + title2)
    
def getConfusionMat(predicted, actual,classcount):
    confusionmatrix = []
    for i in range (classcount):
        confusionmatrix.append([])
        for j in range (classcount):
            confusionmatrix[i].append(0)
    for i in range(0, len(predicted)):
        confusionmatrix[actual[i]][predicted[i]]+=1
    return confusionmatrix

def p_r_c(true_labels,scores):
    plt.figure()
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    plt.plot(recall, precision)
    plt.fill_between(recall, precision, step='post', alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
    plt.title('Precision Recall Curve')
    plt.show()
    
def r_o_c(true_labels,scores):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    plt.plot(fpr, tpr)
    #plt.fill_between(recall, precision, step='post', alpha=0.2,     color='#F59B00')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')
    plt.show()
    
def reports(classifier,train_data,train_labels,train_test_split_ratio=.1):
    scores = []
    print classifier
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
    print ("Per fold Score 5 fold",scores)
    print ("Average Accuracy K Fold: ",scores.mean())
    
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    prob_scores = classifier.predict_proba(X_test)
    r_o_c(y_test,prob_scores[:,1])
    p_r_c(y_test,prob_scores[:,1])
    print ("Test Data Results:")
    print ("Test Accuracy: ",accuracy_score(predicted,y_test))
    X = classification_report(y_test,predicted)
    print (X)
    print "ROC AUC",roc_auc_score(y_true=y_test,y_score=prob_scores[:,1])
    print ("MCC: ",mcc(y_test,predicted))
    heatmap(confusionmat=getConfusionMat(actual=y_test,predicted=predicted,classcount=2))
    print "=============================================================================="

#### for voting classifier
def reports_vote(classifier,train_data,train_labels,train_test_split_ratio=0.1):
   
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
     #heatmap(confusionmat=getConfusionMat(actual=y_test,predicted=predicted,classcount=2))
     print ("MCC: ",mcc(y_test,predicted))
     print "=============================================================================="
     
def reportsData2(classifier,train_data,train_labels,train_test_split_ratio=0.1):
    scores = []
    print classifier
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
    print ("Per fold Score 5 fold",scores)
    print ("Average Accuracy K Fold: ",scores.mean())
    
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    prob_scores = classifier.predict_proba(X_test)
    r_o_c(y_test,prob_scores[:,1])
    p_r_c(y_test,prob_scores[:,1])
    print ("Test Data Results:")
    print ("Test Accuracy: ",accuracy_score(predicted,y_test))
    X = classification_report(y_test,predicted)
    print (X)
    print "ROC AUC",roc_auc_score(y_true=y_test,y_score=prob_scores[:,1])
    print ("MCC: ",mcc(y_test,predicted))
    #heatmap(confusionmat=getConfusionMat(actual=y_test,predicted=predicted,classcount=2))
    print "=============================================================================="
     

    
    
# # Dataset 1 read

completedata = pd.read_csv("./data/data.csv")
completedata.head()
completedata.loc[completedata.diagnosis == 'M', 'diagnosis'] = 1
completedata.loc[completedata.diagnosis == 'B', 'diagnosis'] = 0

train_XGA = completedata[["radius_se","texture_mean","perimeter_mean","perimeter_se","area_se","smoothness_se","compactness_worst","concavity_mean","concavity_worst","concave points_se","concave points_worst","symmetry_mean","symmetry_worst","fractal_dimension_mean"]]
train_YGA = completedata.diagnosis
data1_labels = completedata.diagnosis
data1 = completedata.drop(["id","Unnamed: 32","diagnosis"],axis=1)

train_Y = np.array(data1_labels)
train_X = np.array(data1)
train_YGA = np.array(train_YGA)
train_XGA = np.array(train_XGA)
train_X,train_Y,train_XGA,train_YGA = shuffle(train_X,train_Y,train_XGA,train_YGA)




pca = PCA(n_components=2)
scatter_X = pca.fit_transform(train_X)


classes = ['Malignant','Benign']
colours = ['orange','grey']
recs = []
#plt.xlim([-1000, 1000])
#plt.ylim([-5, 30])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter plot: Undersampling with PCA")
for i in range(0,len(colours)):
    recs.append(pat.Rectangle((0,0),1,1,fc=colours[i]))

for i in range(0,len(scatter_X)):
    if((train_Y[i] == 1)):
        plt.scatter(scatter_X[i][0],scatter_X[i][1],c=colours[0])
    else:
        plt.scatter(scatter_X[i][0],scatter_X[i][1],c=colours[1])
plt.legend(recs,classes,loc='best')
plt.show()

#from sklearn.decomposition import PCA
#pca = PCA(n_components=25)



### features from paper
#print "=============================================================================="
print "\nrotationforest : \n"
reports(rotationforest,train_X,train_Y,0.1)
print "\ndecisiontree : \n"
reports(decisiontree,train_X,train_Y,0.1)
print "\nann : \n"
reports(ann,train_X,train_Y,0.1)
print "\nlogistic regression : \n"
reports(logistic,train_X,train_Y,0.1)
print "\nrandomforest : \n"
reports(randomforest,train_X,train_Y,0.1)
print "\ngausian : \n"
reports(gausian,train_X,train_Y,0.1)
print "\nsvm : \n"
reports_vote(sv,train_X,train_Y,0.1)
print "\nknn : \n"
reports(knn,train_X,train_Y,0.1)
print "\n Voting :\n"
reports_vote(vote,train_X,train_Y,0.1)




####genetic algo on data1
#print "Data 1 :"
#ga = FeatureSelectionGA(rotationforest,train_X,train_Y,cv_split=10)
#pop = ga.generate(100)

 ### report for selected features in paper
print "\nGenetic algo selected features classifiers :"
print "\nrotationforest : \n"
reports(rotationforest,train_XGA,train_YGA,0.1)
print "\ndecisiontree : \n"
reports(decisiontree,train_XGA,train_YGA,0.1)
print "\nann : \n"
reports(ann,train_XGA,train_YGA,0.1)
print "\nlogistic regression : \n"
reports(logistic,train_XGA,train_YGA,0.1)
print "\nrandomforest : \n"
reports(randomforest,train_XGA,train_YGA,0.1)
print "\ngausian : \n"
reports(gausian,train_XGA,train_YGA,0.1)
print "\nsvm : \n"
reports_vote(sv,train_XGA,train_YGA,0.1)
print "\nknn : \n"
reports(knn,train_XGA,train_YGA,0.1)
print "\n Voting :\n"
reports_vote(vote,train_XGA,train_YGA,0.1)







'''Dataset 2 working'''

data2 = np.genfromtxt("./data/breast-cancer-wisconsin.data",delimiter=",")
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

##########pca data visualization
pca = PCA(n_components=2)
scatter_X = pca.fit_transform(train_X2)


classes = ['Malignant','Benign']
colours = ['orange','grey']
recs = []
#plt.xlim([-5, 30])
#plt.ylim([-5, 30])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter plot: Undersampling with PCA")
for i in range(0,len(colours)):
    recs.append(pat.Rectangle((0,0),1,1,fc=colours[i]))

for i in range(0,len(scatter_X)):
    if((train_Y2[i] == 1)):
        plt.scatter(scatter_X[i][0],scatter_X[i][1],c=colours[0])
    else:
        plt.scatter(scatter_X[i][0],scatter_X[i][1],c=colours[1])
plt.legend(recs,classes,loc='best')

plt.show()


# genetic algo
#print "Data 2 :"
#ga = FeatureSelectionGA(rotationforest,train_X2,train_Y2,cv_split=10)
#pop = ga.generate(50,mutxpb=0.4,ngen=6,cxpb=0.6)

print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    
print "\nrotationforest : \n"
reportsData2(rotationforest,train_X2,train_Y2,0.1)
print "\ndecisiontree : \n"
reportsData2(decisiontree,train_X2,train_Y2,0.1)
print "\n ann : \n"
reportsData2(ann,train_X2,train_Y2,0.1)
print "\nlogistic regression : \n"
reportsData2(logistic,train_X2,train_Y2,0.1)
print "\nrandomforest : \n"
reportsData2(randomforest,train_X2,train_Y2,0.1)
print "\ngausian : \n"
reportsData2(gausian,train_X2,train_Y2,0.1)
print "\nsvm : \n"
reports_vote(sv,train_X2,train_Y2,0.1)
print "\nKnn : \n"
reportsData2(knn,train_X2,train_Y2,0.1)
print "\n Voting :\n"
reports_vote(vote,train_X2,train_Y2,0.1)