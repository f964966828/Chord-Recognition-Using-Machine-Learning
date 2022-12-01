from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from utils import load_data


DATA_PATH = "data.json"

X_train, X_test, y_train, y_test = load_data(DATA_PATH)

#Create a Decision tree Classifier
clf = DecisionTreeClassifier()

#Train the model using the training sets
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred, average="macro")
print("~~~ Decision Tree Result ~~~")
print("Accuracy: %.3f | F1 Score: %.3f" % (acc, f1_score))