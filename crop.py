from __future__ import print_function
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
import pickle
warnings.filterwarnings('ignore')

PATH = 'Crop_recommendation.csv'
df = pd.read_csv(PATH)


features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

acc = []
model = []


Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    features, target, test_size=0.2, random_state=2)


DecisionTree = DecisionTreeClassifier(
    criterion="entropy", random_state=2, max_depth=5)

DecisionTree.fit(Xtrain, Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest, predicted_values))


data = np.array([[120, 18, 40, 23.603016, 70, 6.7, 140.91]])
prediction = DecisionTree.predict(data)
print(prediction)

pickle.dump(prediction, open('croppickelfile.pkl', 'wb'))


# model = pickle.load(open('crop_pred.pkl', 'rb'))

# ans = model.predict([[20, 20, 30, 40, 20, 30, 30]])
# print(ans)
