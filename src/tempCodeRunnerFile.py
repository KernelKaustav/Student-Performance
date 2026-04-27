import joblib
import pandas as pd
import shap
model = joblib.load("model.pkl")
df=pd.read_csv("C:/UserData/Kaustav/python projects/student-performance/data/stumath.csv",sep=';')
x=df[['G1','G2','studytime','failures']]
exp = shap.Explainer(model,x)
shap_val=exp(x)
shap.plots.bar(shap_val)
shap.plots.beeswarm(shap_val)