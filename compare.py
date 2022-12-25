import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import get_params, load_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss


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
    y_pred_svc_lin = model_svc_lin.predict(X_test)
    y_pred_svc_rbf = model_svc_rbf.predict(X_test)
    y_predict = [y_pred_knn, y_pred_dt, y_pred_svc_lin, y_pred_svc_rbf]
    

    acc_knn = metrics.accuracy_score(y_test, y_pred_knn)
    acc_dt = metrics.accuracy_score(y_test, y_pred_dt)
    acc_svc_lin = metrics.accuracy_score(y_test, y_pred_svc_lin)
    acc_svc_rbf = metrics.accuracy_score(y_test, y_pred_svc_rbf)
    
    names = ["KNN","Decision Tree", "SVC - Linear Kernal", "SVC - RBF Kernal"]
    accuracy = [acc_knn, acc_dt, acc_svc_lin, acc_svc_rbf]
    
    #print("KNN Accuracy: %.3f" % acc_knn)
    #print("Decision Tree Accuracy: %.3f" % acc_dt)
    #print("SVC - Linear Kernal Accuracy: %.3f" % acc_svc_lin)
    #print("SVC - RBF Kernal Accuracy: %.3f" % acc_svc_rbf)
    
    #plot Algorithm Accuracy Comparison saved as a png file
    max_y_lim = max(accuracy) + 0.01
    min_y_lim = min(accuracy) - 0.01
    fig = plt.figure(1, [10, 6])
    plt.ylim(min_y_lim, max_y_lim)
    plt.bar(x = names , height = accuracy, width=0.4)
    plt.title('Algorithm Accuracy Comparison')
    plt.savefig('Algorithm Accuracy Comparison.png')
    
    target_names = ['A', 'Am', 'Bb', 'Bdim', 'Bm', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G'] 
    #print('KNN_classification_report: \n' + classification_report(y_test, y_pred_knn, target_names=target_names)) 
    #print('Decision Tree_classification_report: \n' + classification_report(y_test, y_pred_dt, target_names=target_names)) 
    #print('SVC - Linear Kernal_classification_report: \n' + classification_report(y_test, y_pred_svc_lin, target_names=target_names)) 
    #print('SVC - RBF Kerna_classification_report: \n' + classification_report(y_test, y_pred_svc_rbf, target_names=target_names)) 
    precision_M = []
    recall_M = []
    f1_score_M = []
    precision_W = []
    recall_W = []
    f1_score_W = []
    
    #classification_report csv file saved as a csv file
    report = metrics.classification_report(y_test, y_pred_knn, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.index = ['A', 'Am', 'Bb', 'Bdim', 'Bm', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G', 'Accuracy', 'Macro avg', 'Weighted avg']
    Macro_avg = df_classification_report.loc['Macro avg'] 
    Weighted_avg = df_classification_report.loc['Weighted avg'] 
    precision_M.append(Macro_avg[0])
    recall_M.append(Macro_avg[1])
    f1_score_M.append(Macro_avg[2])
    precision_W.append(Weighted_avg[0])
    recall_W.append(Weighted_avg[1])
    f1_score_W.append(Weighted_avg[2])
    df_classification_report.to_csv("KNN_classification_report.csv", index = True)
    
    report = metrics.classification_report(y_test, y_pred_dt, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.index = ['A', 'Am', 'Bb', 'Bdim', 'Bm', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G', 'Accuracy', 'Macro avg', 'Weighted avg']
    Macro_avg = df_classification_report.loc['Macro avg'] 
    Weighted_avg = df_classification_report.loc['Weighted avg'] 
    precision_M.append(Macro_avg[0])
    recall_M.append(Macro_avg[1])
    f1_score_M.append(Macro_avg[2])
    precision_W.append(Weighted_avg[0])
    recall_W.append(Weighted_avg[1])
    f1_score_W.append(Weighted_avg[2])
    df_classification_report.to_csv("Decision Tree_classification_report.csv", index = True)
    
    report = metrics.classification_report(y_test, y_pred_svc_lin, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.index = ['A', 'Am', 'Bb', 'Bdim', 'Bm', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G', 'Accuracy', 'Macro avg', 'Weighted avg']
    Macro_avg = df_classification_report.loc['Macro avg'] 
    Weighted_avg = df_classification_report.loc['Weighted avg'] 
    precision_M.append(Macro_avg[0])
    recall_M.append(Macro_avg[1])
    f1_score_M.append(Macro_avg[2])
    precision_W.append(Weighted_avg[0])
    recall_W.append(Weighted_avg[1])
    f1_score_W.append(Weighted_avg[2])
    df_classification_report.to_csv("SVC - Linear Kernal_classification_report.csv", index = True)
    
    report = metrics.classification_report(y_test, y_pred_svc_rbf, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.index = ['A', 'Am', 'Bb', 'Bdim', 'Bm', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G', 'Accuracy', 'Macro avg', 'Weighted avg']
    Macro_avg = df_classification_report.loc['Macro avg'] 
    Weighted_avg = df_classification_report.loc['Weighted avg'] 
    precision_M.append(Macro_avg[0])
    recall_M.append(Macro_avg[1])
    f1_score_M.append(Macro_avg[2])
    precision_W.append(Weighted_avg[0])
    recall_W.append(Weighted_avg[1])
    f1_score_W.append(Weighted_avg[2])
    df_classification_report.to_csv("SVC - RBF_classification_report.csv", index = True)
    
    #plot Algorithm Precision\Recall\F1-score Comparison saved as a png file
    plt.figure(figsize=(12,6)) 
    x = range(4)
    bar1 = plt.bar(x = [i - 0.15 for i in x], height = precision_M, width = 0.3, alpha = 0.8, color = 'r',label = 'Macro avg')                 
    bar2 = plt.bar(x = [i + 0.15 for i in x], height = precision_W ,width = 0.3, alpha = 0.8,color = 'b',label = 'Weighted avg') 
    plt.xticks(x, names)      
    max_y_lim = max(precision_M) + 0.01
    min_y_lim = min(precision_M) - 0.01
    plt.ylim(min_y_lim, max_y_lim)
    plt.title('Algorithm Precision Comparison')     
    plt.legend(loc='best')  
    plt.savefig('Algorithm Precision Comparison.png')
    
    plt.figure(figsize=(12,6)) 
    bar1 = plt.bar(x = [i - 0.15 for i in x], height = recall_M, width = 0.3, alpha = 0.8, color = 'r',label = 'Macro avg')                 
    bar2 = plt.bar(x = [i + 0.15 for i in x], height = recall_W ,width = 0.3, alpha = 0.8,color = 'b',label = 'Weighted avg') 
    plt.xticks(x, names)      
    max_y_lim = max(recall_M) + 0.01
    min_y_lim = min(recall_M) - 0.01
    plt.ylim(min_y_lim, max_y_lim)
    plt.title('Algorithm Recall Comparison')     
    plt.legend(loc='best')  
    plt.savefig('Algorithm Recall Comparison.png')
    
    plt.figure(figsize=(12,6)) 
    bar1 = plt.bar(x = [i - 0.15 for i in x], height = f1_score_M, width = 0.3, alpha = 0.8, color = 'r',label = 'Macro avg')                 
    bar2 = plt.bar(x = [i + 0.15 for i in x], height = f1_score_W ,width = 0.3, alpha = 0.8,color = 'b',label = 'Weighted avg') 
    plt.xticks(x, names)      
    max_y_lim = max(f1_score_M) + 0.01
    min_y_lim = min(f1_score_M) - 0.01
    plt.ylim(min_y_lim, max_y_lim)
    plt.title('Algorithm F1-score Comparison')     
    plt.legend(loc='best')  
    plt.savefig('Algorithm F1-score Comparison.png')
    

    #confusion matrix saved as a png file
    cm = metrics.confusion_matrix(y_true=y_test,  y_pred =  y_pred_knn, labels = range(12))
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', fmt='0.4g')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('KNN_Confusion matrix')
    plt.savefig('KNN_Confusion matrix.png')
    cm = metrics.confusion_matrix(y_true=y_test,  y_pred =  y_pred_dt, labels = range(12))
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', fmt='0.4g')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Decision Tree_Confusion matrix')
    plt.savefig('Decision Tree_Confusion matrix.png')
    cm = metrics.confusion_matrix(y_true=y_test,  y_pred =  y_pred_svc_lin, labels = range(12))
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', fmt='0.4g')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('SVC - Linear Kernal_Confusion matrix')
    plt.savefig('SVC - Linear Kernal_Confusion matrix.png')
    cm = metrics.confusion_matrix(y_true=y_test,  y_pred =  y_pred_svc_rbf, labels = range(12))
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', fmt='0.4g')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('SVC - RBF_Confusion matrix')
    plt.savefig('SVC - RBF_Confusion matrix.png')

    #Log loss
    y_pred_knn = model_knn.predict_proba(X_test)
    log_loss_knn = log_loss(y_test,y_pred_knn)
    #print('KNN_log_loss: ' + str(log_loss_knn))
    y_pred_dt = model_dt.predict_proba(X_test)
    log_loss_dt = log_loss(y_test,y_pred_dt)
    #print('Decision Tree_log_loss: ' + str(log_loss_dt))
    
    #Algorithm Results Comparison saved as a csv file
    print('From Algorithm_Results_Comparison.csv file, we can conclude that KNN has the best-performing in the training process.')
