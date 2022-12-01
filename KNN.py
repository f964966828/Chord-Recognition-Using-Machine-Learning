import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from utils import get_params, load_data

params = get_params()
data_path = params["train_data_path"]
json_name = params["train_json_name"]
n_neighbors = params["n_neighbors"]

json_path = os.path.join(data_path, json_name)

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data(json_path)

    #Create a knn Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    #Train the model using the training sets
    knn.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = knn.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average="macro")
    print("~~~ KNN (n=%d) Result ~~~" % n_neighbors)
    print("Accuracy: %.3f | F1 Score: %.3f" % (acc, f1_score))
