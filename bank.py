import numpy as np
import pandas as pd

df = pd.read_csv("bank.csv")

df.head()

df=df.drop(['ID'],axis=1)

df.head(5)

#Find Null values in the data set:
df.isnull().sum()

#Finding of the duplicate values:
df.duplicated().sum()

#Since all the features in the data set are numerical hence describing the data:
df.describe().transpose()

# Checking class balance for Personal Loan:
df['Personal Loan'].value_counts()

# Class label has imbalanced data, so this feature needs to be re-balanced using upsample method:
# Splitting major & minor class data frames:
df_majority=df[df['Personal Loan']==0]
df_minority=df[df['Personal Loan']==1]

print("Majority calss shape {}".format(df_majority.shape))
print("Minority calss shape {}".format(df_minority.shape))

# Upsampling:
from sklearn.utils import resample
df_minority_upsample=resample(df_minority,n_samples=4520)

df=pd.concat([df_majority,df_minority_upsample])

df['Personal Loan'].value_counts()
df1 = df

# Model Building:
x=df.drop(['Personal Loan'],axis=1)
y=df['Personal Loan']

# Splitting of Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

# Decision Tree Model Prediction
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_pred_base=dt.predict(x_test)

# Finding Accuracy:
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred_base)
print(acc)

# Model validation:
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred_base)

#Classification Report:
clf_report=classification_report(y_test,y_pred_base)
print(clf_report)

# Hyper Parameter Tuning:
from sklearn.model_selection import GridSearchCV
parameters={'criterion':['gini','entropy'],'max_depth':np.arange(1,50),'min_samples_leaf':[1,2,3,6,9,4]}
grid=GridSearchCV(dt,parameters)

model=grid.fit(x_train,y_train)

grid.best_score_

grid.best_params_

clf_best=grid.best_estimator_

clf_best.fit(x_train,y_train)

y_pred_best=clf_best.predict(x_test)

accuracy_score(y_test,y_pred_best)

# Cross Validation:
from sklearn.model_selection import cross_val_score

cross_val=cross_val_score(clf_best,x,y,cv=10)
print(cross_val)

np.mean(cross_val)

# Visualizg the Tree:
from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(16,14))
tree.plot_tree(clf_best)
plt.show()

# For the imbalance data set:
x_imbal=df1.drop(['Personal Loan'],axis=1)
y_imbal=df1['Personal Loan']

x_train_imbal,x_test_imbal,y_train_imbal,y_test_imbal=train_test_split(x_imbal,y_imbal,test_size=0.3)

clf_best.fit(x_train_imbal,y_train_imbal)

y_pred_imbal=clf_best.predict(x_test_imbal)

accuracy_score(y_test_imbal,y_pred_imbal)

