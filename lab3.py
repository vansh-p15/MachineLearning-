import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = load_iris()
a_iris = pd.DataFrame(data.data, columns=data.feature_names)
a_iris["species"] = data.target
print(a_iris.info())
print(a_iris.head())
le = LabelEncoder()
a_iris["species"] = le.fit_transform(a_iris["species"])
X = a_iris.drop(["species"], axis=1)
y = a_iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=data.target_names))
