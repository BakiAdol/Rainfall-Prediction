import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('../input/rainfall-prediction/dataset1.csv')

x = dataset.iloc[:,1:8].values #features
y = dataset.iloc[:,8].values #lable

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25,random_state=0) # train, test split

#make these all data in a range
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

algos = ["Support Vector Classifier","Naive Bayes Classifier","KNeighborsClassifier"]

for i in range(3):
    print("Using ",algos[i]," : ")
    if i==0:
        classifier = SVC()
    elif i==1:
        classifier = GaussianNB()
    else:
        classifier = KNeighborsClassifier(n_neighbors=3)
        
    classifier.fit(x_train,y_train)
    
    #predict result
    y_predict = classifier.predict(x_test)

    #match missmatch
    cm = confusion_matrix(y_test,y_predict)
    print("Confusion Matrix : ")
    print(cm)

    #classification report
    cr = classification_report(y_test, y_predict)
    print("Classification Report : ")
    print(cr)
    print()
    
