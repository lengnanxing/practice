import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv(os.path.join('./input', 'train.csv'))
test = pd.read_csv(os.path.join('./input', 'test.csv'))
print(train['Survived'].groupby(train['Pclass']).mean())
sns.countplot(train['Pclass'],hue=train["Survived"])
plt.show()