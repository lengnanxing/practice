import csv
f0=csv.reader(open("train0.csv","r"))
out=open("train1.csv","a",newline="")
csv_write=csv.writer(out,dialect="excel")
train0=[]
train1=[]
for ft in f0:
    train0.append(ft)
for i in range(len(train0)):
    for j in range(len(train0[i])):
        train0[i][j]=int(float(train0[i][j]))
for i in range(len(train0)):
    csv_write.writerow(train0[i])
print(train0)