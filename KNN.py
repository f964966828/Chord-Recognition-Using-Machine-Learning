from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from utils import load_data

DATA_PATH = "data.json"
N_NEIGHBORDS = 5

X_train, X_test, y_train, y_test = load_data(DATA_PATH)

#Create a knn Classifier
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORDS)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred, average="macro")
print("~~~ KNN Result ~~~")
print("Accuracy: %.3f | F1 Score: %.3f" % (acc, f1_score))
