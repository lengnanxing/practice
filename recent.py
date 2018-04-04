for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
################################################
train = pandas.read_csv('data/titanic_train.csv')
y, X = train['Survived'], train[['Age', 'SibSp', 'Fare']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"Sex",'Age','SibSp'
lr = LogisticRegression()
lr.fit(X_train, y_train)
print accuracy_score(lr.predict(X_test), y_test)

temp=lr.predict(X_test)
out = open("results.csv", "a", newline="")
csv_write = csv.writer(out, dialect="excel")
results=[[""]*2 for  i in range(418)]
print(len(temp))
for i in range(len(temp)):
    results[i][0]=str(892+i)
    results[i][1]=str(temp[i])
for i in range(len(results)):
    csv_write.writerow(results[i])

print(results)



###################
temp=clf.predict(X_test)
out = open("results1.csv", "a", newline="")
csv_write = csv.writer(out, dialect="excel")
results=[[""]*2 for  i in range(418)]
print(len(temp))
for i in range(len(temp)):
    results[i][0]=str(892+i)
    results[i][1]=str(temp[i])
for i in range(len(results)):
    csv_write.writerow(results[i])

print(results)



import  csv
import numpy as np
f0=csv.reader(open("test0.csv","r"))
train0=[]
for tr in f0:
    train0.append(tr)
for i in range(len(train0)):
    print(train0[i][5])
    if float(train0[i][4])>0 or float(train0[i][5])>0:
        train0[i].append(0.0)
    else:
        train0[i].append(1.0)
    if float(train0[i][3])<10:
        train0[i].append(1.0)
    else:
        train0[i].append(0.0)
    if float(train0[i][3])<45 and float(train0[i][3])>18:
        train0[i].append(1.0)
    else:
        train0[i].append(0.0)
out = open("test2.csv", "a", newline="")
csv_write = csv.writer(out, dialect="excel")
for i in range(len(train0)):
    csv_write.writerow(train0[i])
