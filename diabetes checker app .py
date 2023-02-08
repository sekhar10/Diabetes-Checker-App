#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########
##########################################################################
import pandas as pd
from tkinter import messagebox as m
col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima=pd.read_csv("Downloads/pima.csv",names=col_names)
feature=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age']
######################################################################
#dataset cleaning and transformation#######################################
#calculate the median value for a specific column and substitute
#that value everywhere (in the same column) we have zero or null.

# Calculate the median value for BMI
median_bmi = pima['bmi'].median()
##Substitute it in the bmi column of the # dataset where values are 0
pima['bmi'] = pima['bmi'].replace(to_replace=0, value=median_bmi)
##############################################
# Calculate the median value for blood pressure (bp)
median_bp = pima['bp'].median()
# Substitute it in the bp column of the dataset where values are 0
pima['bp'] = pima['bp'].replace(to_replace=0, value=median_bp)
#################################################
# Calculate the median value for plasma glucose concentration(glucose)
median_glucose = pima['glucose'].median()
# Substitute it in the glucose column of the # dataset where values are 0
pima['glucose'] = pima['glucose'].replace(to_replace=0, value=median_glucose)
####################################################################
#calculate the median value of skin thickness
median_skin=pima['skin'].median()
## Substitute it in the skin column of the # dataset where values are 0
pima['skin'] = pima['skin'].replace(to_replace=0, value=median_skin)
######################################################################
#calculate the median value of insulin
median_insulin=pima['insulin'].median()
## Substitute it in the insulin column of the # dataset where values are 0
pima['insulin'] = pima['insulin'].replace(to_replace=0, value=median_insulin)
#print(pima)
####################################################################
#####################################################################
X=pima[feature]
Y=pima.label

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)



#################Apply naive Bayes algorithm
naiveaccuracy=0
def naive():
    from sklearn.naive_bayes import GaussianNB
    global naiveaccuracy
    global NAIVE
    NAIVE=GaussianNB()
    NAIVE.fit(X_train,Y_train)
    Y_pred=NAIVE.predict(X_test)
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
   # print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    #print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    #print("specificity is",specificity)
    naiveaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(naiveaccuracy)+"%")
#print(metrics.accuracy_score(Y_test,Y_pred))
#####Apply Decision Tree Algorithm####    
dtreeaccuracy=0
def dtree():
    from sklearn.tree import DecisionTreeClassifier
    global dtaccuracy
    global DTREE
    DTREE=DecisionTreeClassifier()
    DTREE.fit(X_train,Y_train)
    Y_pred=DTREE.predict(X_test)
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
   # print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    #print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    #print("specificity is",specificity)
    dtaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(dtaccuracy)+"%")
#print(metrics.accuracy_score(Y_test,Y_pred))
############################################################################
#####################APPLY KNN ALGORITHM####################################
from sklearn.neighbors import KNeighborsClassifier
knnaccuracy=0###global

#KNN=KNeighborsClassifier(n_neighbors=5)

def knn():
    '''Apply KNN algorithm to data set'''
    
    global knnaccuracy
    global KNN
    KNN=KNeighborsClassifier(n_neighbors=11)
    KNN.fit(X_train,Y_train)
    Y_pred=KNN.predict(X_test)
    
###############################################################################
                            #confusion matrix
#it is used to check the performance of the algorithm
#find sensitivity,Specificity and Accuracy
##############################################################################
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
   # print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    #print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    #print("specificity is",specificity)
    knnaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(knnaccuracy)+"%")
#print(metrics.accuracy_score(Y_test,Y_pred))

###########################################################################
####################APPLY LOGISTIC REGRESSION#########################
from sklearn.linear_model import LogisticRegression
#LOGREG=LogisticRegression()
logaccuracy=0
def logreg():
    '''Apply Logistic Regression to Data Set'''
    
    global logaccuracy
    global LOGREG
    LOGREG=LogisticRegression(solver='liblinear')
    LOGREG.fit(X_train,Y_train)
    Y_pred=LOGREG.predict(X_test)
    from sklearn import metrics
    confusion=metrics.confusion_matrix(Y_test,Y_pred)
   # print(confusion)
    TP=confusion[1,1]#True Positive
    TN=confusion[0,0]#True Negative
    FP=confusion[0,1]#False Positive
    FN=confusion[1,0]#False Negative
    sensitivity=TP/(TP+FN)#When actual value is positive how many prediction True
    #print("sensitivity is",sensitivity)                      
    specificity=TN/(TN+FP)#When actual value is negative how many prediction True
    #print("specificity is",specificity)
    logaccuracy=round(((TP+TN)/(TP+TN+FP+FN)),2)*100
    m.showinfo(title="Accuracy",message="Accuracy is"+str(logaccuracy)+"%")
#print(metrics.accuracy_score(Y_test,Y_pred))


def compare():
    '''compare the 4 algorithm by using bar'''
    result=[knnaccuracy,logaccuracy,dtaccuracy,naiveaccuracy]
    clf=['KNN','LOG REG','DTREE','NAIVEBAYES']
    import matplotlib.pyplot as plt
    plt.bar(clf,result)
    plt.ylabel("accuracy")
    plt.xlabel("Model")
    plt.title("Model acuracy comparasion")
    plt.show()
##########################################################################
from tkinter import *
import numpy as np
w=Tk()

w.geometry("1200x800")
w.title("Diabetes Prediction System")
w.resizable(0,0)
vpreg=StringVar()
vglucose=StringVar()
vbp=StringVar()
vskin=StringVar()
vinsulin=StringVar()
vbmi=StringVar()
vpedegree=StringVar()
vage=StringVar()
def validate():
    if vpreg.get()=="" or vglucose.get()==""or vbp.get()=="" or vskin.get()==""       or vinsulin.get()=="" or vbmi.get()=="" or vpedegree.get()=="" or vage.get()=="":
        #m.showinfo(title="Wrong Input",message="please fill up all details")
        return
        
        
    
def predict():
    ''' It will predict the status of a new patient'''
    global KNN
    global LOGREG
    #global knnaccuracy
    global logaccuracy
    global sc
    
        
   
    a=float(vpreg.get())
    b=float(vglucose.get())
    c=float(vbp.get())
    d=float(vskin.get())
    e=float(vinsulin.get())
    f=float(vbmi.get())
    g=float(vpedegree.get())
    h=float(vage.get())
        #x=1
    
   
    z=[[a,b,c,d,e,f,g,h]]
    z_trans=sc.transform(z)
    print(z_trans)   
           
  
    l=LOGREG.predict(z_trans)
    if l==0:
        m.showinfo(title="Diabetes Prediction",message="You Have No Diabetes")
    else:
        m.showinfo(title="Diabetes Prediction",message="You have Diabetes or may get soon")
  
    
   
def reset():
    vpreg.set("")
    vglucose.set("")
    vbp.set("")
    vskin.set("")
    vinsulin.set("")
    vbmi.set("")
    vpedegree.set("")
    vage.set("")
    
img=PhotoImage(file='Downloads/diabetesimg.png')
lblimage=Label(w,image=img)
lblimage.grid(row=1,column=1,rowspan=9)
labeltitle=Label(w,text="Enter Your Details!!!!",fg='red',font=('arial',20,'bold'))
labeltitle.grid(row=1,column=2,columnspan=2)
labelpreg=Label(w,text="Pregnant",font=('arial',20,'bold'))
labelpreg.grid(row=2,column=2)
entrypreg=Entry(w,font=('arial',20,'bold'),textvariable=vpreg)
entrypreg.grid(row=2,column=3)
labelglucose=Label(w,text="Glucose",font=('arial',20,'bold'))
labelglucose.grid(row=3,column=2)
entryglucose=Entry(w,font=('arial',20,'bold'),textvariable=vglucose)
entryglucose.grid(row=3,column=3)
labelbp=Label(w,text="Blood Pressure",font=('arial',20,'bold'))
labelbp.grid(row=4,column=2)
entrybp=Entry(w,font=('arial',20,'bold'),textvariable=vbp)
entrybp.grid(row=4,column=3)
labelskin=Label(w,text="Skin fold thickness",font=('arial',20,'bold'))
labelskin.grid(row=5,column=2)
entryskin=Entry(w,font=('arial',20,'bold'),textvariable=vskin)
entryskin.grid(row=5,column=3)
labelinsulin=Label(w,text="Insulin",font=('arial',20,'bold'))
labelinsulin.grid(row=6,column=2)
entryinsulin=Entry(w,font=('arial',20,'bold'),textvariable=vinsulin)
entryinsulin.grid(row=6,column=3)
labelbmi=Label(w,text="Body Mass Index",font=('arial',20,'bold'))
labelbmi.grid(row=7,column=2)
entrybmi=Entry(w,font=('arial',20,'bold'),textvariable=vbmi)
entrybmi.grid(row=7,column=3)
#
labelpedegree=Label(w,text="Pedigree",font=('arial',20,'bold'))
labelpedegree.grid(row=8,column=2)
entrypedegree=Entry(w,font=('arial',20,'bold'),textvariable=vpedegree)
entrypedegree.grid(row=8,column=3)
#
labelage=Label(w,text="Age",font=('arial',20,'bold'))
labelage.grid(row=9,column=2)
entryage=Entry(w,font=('arial',20,'bold'),textvariable=vage)
entryage.grid(row=9,column=3)
#########################################################################
#########################################################################
btnpredict=Button(w,text="Predict",bg='yellow',width=10,relief='groove',font=('arial',20,'bold'),fg='green',command=predict)
btnpredict.grid(row=10,column=2)
btnreset=Button(w,text="Reset",bg='yellow',width=10,relief='groove',font=('arial',20,'bold'),fg='green',command=reset)
btnreset.grid(row=10,column=3)
btnknn=Button(w,text="  KNN  ",bg='cyan',font=('arial',20,'bold'),command=knn)
btnknn.grid(row=10,column=1)
btndt=Button(w,text="Decision Tree",bg='cyan',font=('arial',20,'bold'),command=dtree)
btndt.grid(row=11,column=1)
btndt=Button(w,text="Naive Bayes",bg='cyan',font=('arial',20,'bold'),command=naive)
btndt.grid(row=12,column=1)
btnlogreg=Button(w,text="Logistic Regression",bg='cyan',font=('arial',20,'bold'),command=logreg)
btnlogreg.grid(row=13,column=1)
btncompare=Button(w,text="Compare",bg='cyan',border=10,font=('arial',20,'bold'),command=compare)
btncompare.grid(row=14,column=1)
############################################################################
w.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




