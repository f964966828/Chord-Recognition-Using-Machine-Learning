import os
from sklearn import metrics
from sklearn.svm import SVC
from utils import get_params, load_data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

params = get_params()
image_path = params["image_path"]
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

    #Using PCA to reduce X to 2 dimension
    pca_SVM = PCA(n_components=2)
    pca_SVM.fit(X_train)

    pca_X_train = pca_SVM.transform(X_train)
    pca_X_test = pca_SVM.transform(X_test)
    
    x_te, y_te = pca_X_test[:, 0], pca_X_test[:, 1]

    #meshgrid
    h = .01
    x_min, x_max = pca_X_train[:, 0].min() - 0.1, pca_X_train[:, 0].max() + 0.1
    y_min, y_max = pca_X_train[:, 1].min() - 0.1, pca_X_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    #Create another SVM Classifier to predict xx yy
    svclassifier_lin2d = SVC(kernel='linear')
    svclassifier_rbf2d = SVC(kernel='rbf')
    #Train the model using the reduced training sets
    SVM_lin2d = svclassifier_lin2d.fit(pca_X_train, y_train)
    SVM_rbf2d = svclassifier_rbf2d.fit(pca_X_train, y_train)
    #Predict Z using xx yy
    Z_lin = SVM_lin2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_rbf = SVM_rbf2d.predict(np.c_[xx.ravel(), yy.ravel()])

    colors = {0: 'red',
            1: '#FFC1E0',
            2: 'green', 
            3: '#4EFEB3', 
            4: '#82D900',  
            5:'purple',
            6:'blue',
            7:'#7373B9',
            8:'#ADADAD',
            9:'#9999CC',
            10:'orange',
            11:'yellow'}
    
    names = {0: 'A',
            1: 'Am', 
            2: 'Bb', 
            3: 'Bdim', 
            4: 'Bm',
            5:'C',
            6:'D',
            7:'Dm',
            8:'E',
            9:'Em',
            10:'F',
            11:'G'}

    df_te_lin = pd.DataFrame({'x_te': x_te, 'y_te':y_te, 'label_te_lin':y_pred_lin}) 
    df_te_rbf = pd.DataFrame({'x_te': x_te, 'y_te':y_te, 'label_te_rbf':y_pred_rbf}) 

    groups_te_lin = df_te_lin.groupby('label_te_lin')
    groups_te_rbf = df_te_rbf.groupby('label_te_rbf')

    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(121) 
    ax2 = fig.add_subplot(122) 

    # Put the result into a color plot
    Z_lin = Z_lin.reshape(xx.shape)
    Z_rbf = Z_rbf.reshape(xx.shape)
     
    #####################
    cmap_idx_lin = [idx for idx in colors.keys() if idx in Z_lin]
    cmap_idx_rbf = [idx for idx in colors.keys() if idx in Z_rbf]
    cmap_lin = ListedColormap([colors[idx] for idx in cmap_idx_lin])
    cmap_rbf = ListedColormap([colors[idx] for idx in cmap_idx_rbf])

    plot_map_lin = dict()
    plot_map_rbf = dict()
    for i, idx in enumerate(cmap_idx_lin):
        plot_map_lin[idx] = i
    for i, idx in enumerate(cmap_idx_rbf):
        plot_map_rbf[idx] = i
    
    plot_Z_lin = [[plot_map_lin[z] for z in row] for row in Z_lin]
    plot_Z_rbf = [[plot_map_rbf[z] for z in row] for row in Z_rbf]

    ax1.pcolormesh(xx, yy, plot_Z_lin, cmap=cmap_lin, alpha=0.7) #感覺是這個cmap有錯(?
    ax2.pcolormesh(xx, yy, plot_Z_rbf, cmap=cmap_rbf, alpha=0.7)
    #####################

    for name, group in groups_te_lin:
        ax1.scatter(group.x_te, group.y_te, linewidths=0.5, edgecolors = 'black', s=20,
                color=colors[name], label=names[name])
        ax1.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
        ax1.tick_params(axis='y',which='both',left='off',top='off',labelleft='off')

    ax1.legend()
    ax1.set_title(f"SVM(Linear)\n Accuracy:{round(acc_lin, 3)}")

    for name, group in groups_te_rbf:
        ax2.scatter(group.x_te, group.y_te, linewidths=0.5, edgecolors = 'black',s=20,
                color=colors[name],label=names[name])
        ax2.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
        ax2.tick_params(axis='y',which='both',left='off',top='off',labelleft='off')

    ax2.legend()
    ax2.set_title(f"SVM(RBF)\n Accuracy:{round(acc_rbf, 3)}")

    plt.savefig(os.path.join(image_path, 'SVM_visualization.png'))
