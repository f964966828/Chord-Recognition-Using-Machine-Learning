import os
from sklearn import metrics
from sklearn.svm import SVC
from utils import get_params, load_data

params = get_params()
data_path = params["train_data_path"]
json_name = params["train_json_name"]
json_path = os.path.join(data_path, json_name)

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = load_data(json_path)

    #Create a SVM Classifier
    svclassifier_lin = SVC(kernel='linear')
    svclassifier_rbf = SVC(kernel='rbf')
    #Train the model using the training sets
    svclassifier_lin.fit(X_train, y_train)
    svclassifier_rbf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred_lin = svclassifier_lin.predict(X_test)
    y_pred_rbf = svclassifier_rbf.predict(X_test)

    acc_lin = metrics.accuracy_score(y_test, y_pred_lin)
    acc_rbf = metrics.accuracy_score(y_test, y_pred_rbf)
    f1_score_lin = metrics.f1_score(y_test, y_pred_lin, average="macro")
    f1_score_rbf = metrics.f1_score(y_test, y_pred_rbf, average="macro")
    print("~~~ SVM - Linear Kernal Result ~~~")
    print("Accuracy: %.3f | F1 Score: %.3f" % (acc_lin, f1_score_lin))
    print("~~~ SVM - RBF Kernal Result ~~~")
    print("Accuracy: %.3f | F1 Score: %.3f" % (acc_rbf, f1_score_rbf))
