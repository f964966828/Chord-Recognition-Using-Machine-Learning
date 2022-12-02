import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import get_params, load_data

params = get_params()
data_path = params["train_data_path"]
json_name = params["train_json_name"]
json_path = os.path.join(data_path, json_name)

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = load_data(json_path)

    #Create Classifier
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_dt=DecisionTreeClassifier()
    model_svc_lin=SVC(kernel='linear')
    model_svc_rbf=SVC(kernel='rbf')

    #Train the model using the training sets
    model_knn.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)
    model_svc_lin.fit(X_train, y_train)
    model_svc_rbf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred_knn = model_knn.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)
    y_pred_svm_lin = model_svc_lin.predict(X_test)
    y_pred_svm_rbf = model_svc_rbf.predict(X_test)

    acc_knn = metrics.accuracy_score(y_test, y_pred_knn)
    acc_dt = metrics.accuracy_score(y_test, y_pred_dt)
    acc_svm_lin = metrics.accuracy_score(y_test, y_pred_svm_lin)
    acc_svm_rbf = metrics.accuracy_score(y_test, y_pred_svm_rbf)
    
    print("KNN Accuracy: %.3f" % acc_knn)
    print("Decision Tree Accuracy: %.3f" % acc_dt)
    print("SVM - Linear Kernal Accuracy: %.3f" % acc_svm_lin)
    print("SVM - RBF Kernal Accuracy: %.3f" % acc_svm_rbf)
