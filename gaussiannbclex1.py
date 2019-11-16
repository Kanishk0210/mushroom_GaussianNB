import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split

data=pd.read_csv("/home/kanishk/Downloads/mushrooms.csv")
c=pd.Series(data['class'])
f1=pd.DataFrame(data)
f2=f1.drop('class',axis=1)
f3=np.asanyarray(f2).transpose()
le=preprocessing.LabelEncoder()
cf=le.fit_transform(c)
li=[]
for i in range(0,22):
    c1=le.fit_transform(f3[i])
    li.append(c1)
f4=np.asanyarray(li).reshape(22,8124)
ff=f4.transpose()

print(cf)
x_train,x_test,y_train,y_test=train_test_split(ff,cf,test_size=0.3)
model=GaussianNB().fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print("Accuracy of model :",metrics.accuracy_score(y_test,y_pred),model.score(x_train,y_train))
