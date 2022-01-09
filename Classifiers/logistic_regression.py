import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sc = StandardScaler()
df = pd.read_csv('../Dataset/BrainTumor.csv', index_col=0)
df = df[df['Mean'] > 0]
# for x in df.columns:
#     if(x == "Class"):
#         continue
#     df[x] = (df[x]/df[x].max()) 
# print(len(df[df['Class'] == 0]))
# exit()
y = df['Class']
print(df)
df.drop('Class', inplace=True, axis=1)
X = df
print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

# cols = df.columns.drop(['Class'])
# formula = 'Class ~ ' + ' + '.join(cols)
# model = smf.glm(formula=formula, data=X_train, family=sm.families.Binomial())
# logistic_fit = model.fit()
# print(logistic_fit.summary())
# predictions = logistic_fit.predict(X_test)
# print(logisticRegr.summary())
predictions = logisticRegr.predict(X_test)
print(predictions[1:6])

predictions_nominal = [ "1" if x == 1.0 else "0" for x in predictions]
print(predictions_nominal[1:6])

print(classification_report(y_test.astype(int).astype(str), predictions_nominal, digits=3))
cfm = confusion_matrix(y_test.astype(int).astype(str), predictions_nominal)
true_negative = int(cfm[0][0])
false_positive = int(cfm[0][1])
false_negative = int(cfm[1][0])
true_positive = int(cfm[1][1])

print('Confusion Matrix: \n', cfm, '\n')
print('True Negative:', true_negative)
print('False Positive:', false_positive)
print('False Negative:', false_negative)
print('True Positive:', true_positive)
print('')
print('Precision TN:', round(true_negative/(true_negative+false_negative)*100),"%")
print('Recall TN:', round(true_negative/(true_negative+false_positive)*100),"%")
print('Precision TP:', round(true_positive/(true_positive+false_positive)*100),"%")
print('Recall TP:', round(true_positive/(true_positive+false_negative)*100),"%")
print('Accuracy : ', 
      round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')


cm = confusion_matrix(y_test, predictions)
f = sns.heatmap(cm, annot=True, fmt='d')
plt.show()