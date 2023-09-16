import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"/Users/Garima/Desktop/Python/Titanic/titanic.csv")
print(df.head())
print(df.tail())
print(df.info())
print("Missing values:\n", df.isnull().sum())
df['Age'].fillna(df['Age'].mean(), inplace=True)
i=df.drop(['Survived','Name','Cabin'],axis='columns')
t=df['Survived']
Sex_le=LabelEncoder()
#Ticket_le=LabelEncoder()
#Embarked_le=LabelEncoder()
i['Sex']=Sex_le.fit_transform(i['Sex'])
#i['Ticket']=Ticket_le.fit_transform(i['Ticket'])
#i['Embarked']=Embarked_le.fit_transform(i['Embarked'])
i = pd.get_dummies(i, columns=['Ticket', 'Embarked'], drop_first=True)
print(i)
model=DecisionTreeClassifier()
X_train,X_test,y_train,y_test=train_test_split(i,t,test_size=0.2,random_state=23)
model.fit(X_train,y_train)
pre=model.predict(X_test)
print("Accuaracy:",model.score(X_test,y_test)*100)
pre=model.predict(X_test)
cm=confusion_matrix(y_test,pre)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True)
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.show()
pr=precision_score(y_test,pre)
print("Precision:")
print(pr)
f1 = f1_score(y_test, pre)
print("F1 Score:", f1)
