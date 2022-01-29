import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score


# Data processing module.
def load_data(data_index, pre_index, test_size):
    """
    data_index:（1,2,3）
    pre_index: （1,2,3）
    test_size:  (0-1)
    """
    if (data_index == 103):
        df = pd.read_csv("dataset.csv")
        X = df.values[:, :13]
        y = df.values[:, 12 + pre_index]
    else:
        print("Data selection error ！")
        return 0

    poly = PolynomialFeatures(2, include_bias=0)
    # implement polynomial degree=2
    X = poly.fit_transform(X)
    X_n = X
    y_n = y

    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_n, y_n,
                                                                test_size=test_size)  # These variables are before gaussian noise.

    for i in range(5):  # -------add gaussian noise.
        X_noise = np.random.normal(0, 1, np.shape(X)) * 0.01
        X_n = np.r_[X_n, X + X_noise]
        y_n = np.r_[y_n, y]

    X_train, X_test, y_train, y_test = train_test_split(X_n, y_n,
                                                        test_size=test_size)  # These vairables are after gaussian noise.

    return X_train, X_test, y_train, y_test, X, y, X_test_o, y_test_o, X_train_o, y_train_o

# Data visualization module.
def show_2(model,model_name,y_pred1,y_pred2,y_test1,y_test2):
    y_pred_all = np.hstack((y_pred1,y_pred2))
    y_test_all = np.hstack((y_test1,y_test2))

    #plt.style.use("ggplot")
    method=model_name
    plt.figure(dpi=200,figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(y_pred1)),y_pred1, "ro-", label="Predict value") # Here, change y_pred_all to y_pred1 will make plot much cleaner. Highly recommended.
    plt.plot(np.arange(len(y_test1)),y_test1, "go-", label="True value")
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Value')
    y_pred_all = np.hstack((y_pred1,y_pred2))
    y_test_all = np.hstack((y_test1,y_test2))
    RMSE1 = (mean_squared_error(y_pred_all,y_test_all))**0.5
    RMSE2 = (mean_squared_error(y_pred1,y_test1))**0.5
    plt.title(f"{method} RMSE:{('%.5f'%RMSE1)}eV")
    print("Test set RMSE："+str((mean_squared_error(y_pred1,y_test1))**0.5))
    plt.legend(loc='upper left')
    plt.xticks(np.arange(0,len(y_test1),5))

    # Second plot.
    plt.subplot(1,2,2)
    plt.scatter(y_pred1,y_test1,marker='x',color='red',s = 10,label = "Test set sample")
    plt.scatter(y_pred2,y_test2,marker='o',color='green',s = 5,label = "Training set sample")
    plt.legend(loc='upper left')
    plt.xlabel('ML predicted')
    plt.ylabel('DFT calculated')
    plt.title(r'$R^2$'+' '+str(r2_score(y_pred_all,y_test_all)))
    print("Test set R2："+str(r2_score(y_pred1,y_test1)))
    plt.savefig(str(model)+".png")

#model
models = []
model_names = ["BayesRidge Regression"]
models.append(linear_model.BayesianRidge(n_iter=300, tol=10^-3))
X_train,X_test,y_train,y_test,Original_x,Origial_y,X_test_o,y_test_o,X_train_o,y_train_o = load_data(data_index=103,pre_index=1,test_size=0.2)
j = 0
for i,model in enumerate(models):
    model.fit(X_train,y_train)
    y_pred1 = model.predict(X_test_o)
    y_pred2 = model.predict(X_train_o)
    y_test1 = y_test_o
    y_test2 = y_train_o
    show_2(model,model_names[j],y_pred1,y_pred2,y_test1,y_test2)
    j+=1

#models
models = []
model_names = ["BayesRidge Regression"]

models.append(linear_model.BayesianRidge(n_iter=300, tol=10^-3))
# 500 times average statistics.

for j, model in enumerate(models):
    All_R2_temp = []
    Train_R2_temp = []
    Test_R2_temp = []
    All_RMSE_temp = []
    Train_RMSE_temp = []
    Test_RMSE_temp = []

    for i in range(500):
        X_train, X_test, y_train, y_test, Original_x, Origial_y, X_test_o, y_test_o, X_train_o, y_train_o = load_data(
            data_index=103, pre_index=1, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred1 = model.predict(X_test_o)
        y_pred2 = model.predict(X_train_o)
        y_test1 = y_test_o
        y_test2 = y_train_o
        y_pred1 = model.predict(X_test_o)
        y_pred2 = model.predict(X_train_o)
        y_pred_all = np.hstack((y_pred1, y_pred2))
        y_test_all = np.hstack((y_test1, y_test2))
        All_R2_temp.append(r2_score(y_pred_all, y_test_all))
        Train_R2_temp.append(r2_score(y_pred2, y_test2))
        Test_R2_temp.append(r2_score(y_pred1, y_test1))
        All_RMSE_temp.append(mean_squared_error(y_pred_all, y_test_all) ** 0.5)
        Train_RMSE_temp.append(mean_squared_error(y_pred2, y_test2) ** 0.5)
        Test_RMSE_temp.append(mean_squared_error(y_pred1, y_test1) ** 0.5)
    Test_R21 = Test_R2_temp
    All_RMSE = (sum(All_RMSE_temp) / 500)
    All_R2 = sum(All_R2_temp) / 500
    Train_RMSE = (sum(Train_RMSE_temp) / 500)
    Train_R2 = sum(Train_R2_temp) / 500
    Test_RMSE = (sum(Test_RMSE_temp) / 500)
    Test_R2 = sum(Test_R2_temp) / 500

    print("model: " + str(model))

    print("All data set RMSE:{}".format(All_RMSE))
    print("All data set R2:{}".format(All_R2))

    print("Training set RMSE:{}".format(Train_RMSE))
    print("Training set R2:{}".format(Train_R2))

    print("Test set RMSE:{}".format(Test_RMSE))
    print("Test set R2:{}".format(Test_R2))
    print("\n")