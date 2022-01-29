import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

#Prediction
file = pd.read_csv('dataset.csv')

X = file.iloc[:,:13]
y = file.iloc[:,13]

for i in range(5):#-------add gaussian noise.
    X_noise=np.random.normal(0,1,np.shape(X))*0.01
    X_n=np.r_[X,X+X_noise]
    y_n=np.r_[y,y]

#PolynomialFeatures
poly = PolynomialFeatures(2,include_bias=0)
X_n=poly.fit_transform(X_n)

X_train,X_test,y_train,y_test = train_test_split(X_n,y_n,test_size=0.2)
#GBR
model = GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05)

model.fit(X_train,y_train)

predict_original_X = pd.read_csv('predicted_set.csv').iloc[:,:13]
predict_original_X = poly.fit_transform(predict_original_X)
predict_result = model.predict(predict_original_X)
#Prediction results
np.savetxt('results.csv', predict_result, delimiter=",")