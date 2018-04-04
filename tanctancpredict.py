import keras
import random
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import csv
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

train = pandas.read_csv('train2.csv')
test=pandas.read_csv("test2.csv")
y, X = train['Survived'], train[["Pclass","Fare"]].fillna(0)
x_test=test[["Age","Fare","SibSp","Parch"]].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gbdt_train_X,gbdt_train_y=train[["Age","Fare","SibSp","Parch"]],train["Survived"]
##clf =svm.SVC(gamma=0.001,C=100)
##,"Sex",'Age','SibSp'"Pclass"]
##clf=RandomForestClassifier(100)
##clf.fit(X_train, y_train)
##print(accuracy_score(
##    clf.predict(X_test),y_test))
##temp=clf.predict(x_test)

##DataSet Conduct
gbr=GradientBoostingRegressor()#x[i0]为训练样本输入，y[i0]为训练样本输出
gbr.fit(gbdt_train_X, gbdt_train_y)#训练GBDT模型
enc = OneHotEncoder()
enc.fit(gbr.apply(gbdt_train_X))#将位置码转化为01码
new_feature_train=enc.transform(gbr.apply(gbdt_train_X))
new_feature_train=new_feature_train.toarray()
##For Adjust
print(len(new_feature_train[0]))
enc1= OneHotEncoder()
enc1.fit(train[["Pclass","Sex","IsAlone","IsChild","IsStrong"]])
new_feature_train1=enc1.transform(train[["Pclass","Sex","IsAlone","IsChild","IsStrong"]])
new_feature_train1=new_feature_train1.toarray()
new_train=np.concatenate([new_feature_train1,new_feature_train],axis=1)

new_feature_test=enc.transform(gbr.apply(x_test))
new_feature_test=new_feature_test.toarray()
##For Adjust
print(len(new_feature_test[0]))
new_feature_test1=enc1.transform(test[["Pclass","Sex","IsAlone","IsChild","IsStrong"]])
new_feature_test1=new_feature_test1.toarray()
new_test=np.concatenate([new_feature_test1,new_feature_test],axis=1)
##print(len(new_train[0]),len(new_test[0]))
y = keras.utils.to_categorical(y,num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(new_train, y, test_size=0.1, random_state=42)
##y_train=pandas.get_dummies(y_train)
##X_train["Pclass"]=pandas.get_dummies(X_train["Pclass"])
##print(X_train)
##gbdt训练得到新特征
##For Adjust
print(len(y_train[0]),y_train[0])
##model setting
model = Sequential()
model.add(Dense(64, input_dim=753, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          epochs=40,
          batch_size=20)
print(model.evaluate(X_test, y_test))
p1=model.predict(new_test)

p2=[]
for i in range(len(p1)):
    if p1[i][0]>p1[i][1]:
        p2.append(0)
    else:
        p2.append(1)
print(p2)
"""""""""
##将预测结果写入到.csv文件
temp=p2
out = open("results5.csv", "a", newline="")
csv_write = csv.writer(out, dialect="excel")
results=[[""]*2 for  i in range(418)]
print(len(temp))

for i in range(len(temp)):
    results[i][0]=str(892+i)
    results[i][1]=str(int(temp[i]))
for i in range(len(results)):
    csv_write.writerow(results[i])


"""""""""""
