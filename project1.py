import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("archive/data.csv")
print(df.columns)
print(df.tail())
new_df = df.drop(["date","country","street"] , axis = 1)
print(new_df)
print(df.isnull().sum())
new_df = pd.get_dummies(new_df,columns=["city" , "statezip"] , drop_first= True)
new_df = new_df[new_df["price"]>0]
y = np.log(new_df["price"])
new_df["yr_renovated"] = new_df["yr_renovated"].where(new_df["yr_renovated"]!=0,new_df["yr_built"])
X = new_df.drop([ "price" , "floors" , "waterfront"] , axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2 , random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

Kn = KNeighborsRegressor(n_neighbors=3)
Kn.fit(X_train_sc,y_train)
y_predict = Kn.predict(X_test_sc)
print(np.sqrt(mean_squared_error(y_test, y_predict)))

model = LinearRegression()
model.fit(X_train_sc , y_train)
y_li_predict = model.predict(X_test_sc)
print(np.sqrt(mean_squared_error(y_test,y_li_predict)))

model_2 = tree.DecisionTreeRegressor(max_depth=5)
model_2.fit(X_train_sc,y_train)
y_linear_predict = model_2.predict(X_test_sc)
print(np.sqrt(mean_squared_error(y_test,y_linear_predict)))

model_3 = RandomForestRegressor(n_estimators=200,max_features='sqrt',max_depth=15 ,min_samples_leaf=2,min_samples_split=5)
model_3.fit(X_train_sc , y_train)
y_rf_predict = model_3.predict(X_test_sc)
print(np.sqrt(mean_squared_error(y_test,y_rf_predict)))

print("KNN r2 score",r2_score(y_test,y_predict))
print("linear regression  r2 score",r2_score(y_test,y_li_predict))
print("decision tree r2 score",r2_score(y_test,y_linear_predict))
print("RF r2 score",r2_score(y_test,y_rf_predict))
print((new_df['price']==0).sum())







