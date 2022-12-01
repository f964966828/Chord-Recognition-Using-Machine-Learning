from sklearn import metrics
from sklearn.svm import SVC
from utils import load_data


DATA_PATH = "data.json"

X_train, X_test, y_train, y_test = load_data(DATA_PATH)

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
