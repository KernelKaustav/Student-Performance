import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib

df =  pd.read_csv("C:/UserData/Kaustav/python projects/student-performance/data/stumath.csv",sep=';')
x = df[['G1' , 'G2' , 'studytime','failures']]
y = df['G3']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33,random_state=40 )

#------------------------------------------------------------------------
model1=DecisionTreeRegressor(random_state=40)
pg = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid = GridSearchCV(
    estimator=model1,
    param_grid=pg,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid.fit(x_train , y_train)
best_model = grid.best_estimator_
print("BEST PARAMETERS : ",grid.best_params_)
pred1 = best_model.predict(x_test)
print("MSE:", mean_squared_error(y_test, pred1))
print("R2:", r2_score(y_test, pred1))

#---------------------------------------------------------------------------

model2 = RandomForestRegressor(random_state=40)

pg = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_rf = GridSearchCV(
    model2,
    param_grid=pg,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_rf.fit(x_train, y_train)

best_rf = grid_rf.best_estimator_

pred2 = best_rf.predict(x_test)

print("Best RF Params:", grid_rf.best_params_)
print("RF MSE:", mean_squared_error(y_test, pred2))
print("RF R2:", r2_score(y_test, pred2))

#-------------------------------------------------------------------------------
if r2_score(y_test, pred2) > r2_score(y_test, pred1):
    final_model = best_rf
    model_name = "Random Forest"
    final_pred = pred2
else:
    final_model = best_model
    model_name = "Decision Tree"
    final_pred = pred1

print("\nFinal Model Selected:", model_name)

imp = final_model.feature_importances_
features = x.columns
plt.barh(features, imp)
plt.title("Feature Importance")
plt.show()

joblib.dump(final_model, "model.pkl")