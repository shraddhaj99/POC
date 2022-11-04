import seaborn as sns
import matplotlib.pyplot as plt
from bank import df
import numpy as np

df.columns

col=['Age', 'Experience', 'Income', 'CCAvg','Mortgage']

i=3
j=0
plt.figure(figsize=(14,12))
for k in col :
    plt.subplot(i,i,i*(j+1)//i)
    sns.distplot(df[k])
    j=j+1
plt.show()

# Replacing negative experience values with the median value in the Experience column:
negexp=df[df['Experience']<0]

negexp['Experience'].value_counts()

negval=[-3, -2, -1]

for i in negval:
    df['Experience']=df['Experience'].replace(negval,np.median(df['Experience']))

df['Experience'].describe()

# Finding Corelation between the features:
cor=df.corr()

# Heatmap for Corelation:
plt.figure(figsize=(10,8))
plt.title("Corelation Plot")
sns.heatmap(cor,annot=True)
plt.show()

plt.figure(figsize=(10,8))
plt.title("Scatter plot for Experience & Age")
sns.scatterplot(x='Age',y='Experience', hue='Personal Loan', data=df)
plt.show()

df=df.drop(['Experience'],axis=1)

# Plotting Scatter plot for multivariate features:
col=['Income','CCAvg','Mortgage']
plt.figure(figsize=(14,12))
j=3
k=0
for i in col:
    plt.subplot(1,j,j*(k+1)//j)
    sns.scatterplot(x='Age',y=i,hue='Personal Loan', data=df)
    k=k+1
plt.show()

# Plotting Counts plot for Categorical features:
col=['Securities Account','CD Account','Online','CreditCard']
plt.figure(figsize=(14,12))
j=2
k=0
for i in col:
    plt.subplot(2,j,j*(k+1)//j)
    sns.countplot(x=i,hue='Personal Loan', data=df)
    k=k+1
    plt.grid(True)
plt.show()

df.columns

plt.figure(figsize=(9,7))
sns.boxplot(x='Family',y='Income',hue='Personal Loan', data=df)
plt.show()

plt.figure(figsize=(12,10))
sns.boxplot(x='Education',y='CCAvg',hue='Personal Loan', data=df)
plt.show()

df.columns

df=df.drop(['ZIP Code'],axis=1)

df1=df

df1['Personal Loan'].value_counts()

df.head(5)