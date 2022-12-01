import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import get_params, load_data

params = get_params()
train_data_path = params["train_data_path"]
train_json_name = params["train_json_name"]
test_data_path = params["test_data_path"]
test_json_name = params["test_json_name"]

train_json_path = os.path.join(train_data_path, train_json_name)
test_json_path = os.path.join(test_data_path, test_json_name)

if __name__ == "__main__":

    X, y, m = load_data(train_json_path, return_mapping=True)
    X_test, _, _ = load_data(test_json_path, return_mapping=True)

    #Create a knn Classifier
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_dt=DecisionTreeClassifier()
    model_svc_lin=SVC(kernel='linear')
    model_svc_rbf=SVC(kernel='rbf')

    #Train the model using the training sets
    model_knn.fit(X, y)
    model_dt.fit(X, y)
    model_svc_lin.fit(X, y)
    model_svc_rbf.fit(X, y)

    #Predict the response for test dataset
    y_pred_knn = model_knn.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)
    y_pred_svm_lin = model_svc_lin.predict(X_test)
    y_pred_svm_rbf = model_svc_rbf.predict(X_test)

    #prediction of where_did:
    print("KNN: ")
    for i in range(len(X_test)):
        print(m[y_pred_knn[i]],end=' ' )
    print("\nDecision tree: ")
    for i in range(len(X_test)):
        print(m[y_pred_dt[i]],end=' ' )
    print("\nSVM rbf: ")
    for i in range(len(X_test)):
        print(m[y_pred_svm_rbf[i]],end=' ' )
    print("\nSVM linear: ")
    for i in range(len(X_test)):
        print(m[y_pred_svm_lin[i]],end=' ' )
    print("\nReal chords: ")
    print("em g em g em em em g em em g g em g g g em g g g em em g em em g g em em g g")
