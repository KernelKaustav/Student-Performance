This project predicts a student's final grade (G3) using machine learning models based on academic and behavioral features such as previous grades, study time, and failures.

AIM : 
To build an end-to-end machine learning pipeline that:
- Analyzes student performance data  
- Trains predictive models  
- Selects the best model using evaluation metrics  
- Explains predictions using SHAP  
- Deploys the model through a web application

DATASOURCE
- Source: UCI Student Performance Dataset  
- Features used:
  - G1 (First period grade)
  - G2 (Second period grade)
  - studytime
  - failures  
- Target:
  - G3 (Final grade)
 
  EDA
- Identified **G2 and G1 as the strongest predictors** of final performance  
- Observed strong correlation between G2 and G3  
- Analyzed feature distributions and relationships using visualizations  

MODELS USED FOR TRAINING
- Decision Tree Regressor  
- Random Forest Regressor

MODEL OPTIMIZATION(Hyperparameter tuning)

- Applied **GridSearchCV with 5-fold cross-validation** to tune hyperparameters  
- Each model was trained and validated across multiple data splits to ensure robustness  
- Evaluated performance using average R² score across folds  
- Tuned parameters:
  - max_depth  
  - min_samples_split  
  - min_samples_leaf  
  - n_estimators (Random Forest)
 
 RESULTS (on TESTING SET)
  - DECISION TREE
    R2 Score = 0.79
    MSE = 4.0
  - Random Forest
    R2 Score = 0.81
    MSE = 3.9
  SO WE CONSIDERED RANDOM FOREST AS OUR BEST MODEL .

MODEL EXPLANATION
SHAP was used to understand model predictions:

- **G2 (second period grade)** has the highest impact on final grade  
- **G1** has moderate influence  
- **Failures** negatively affect performance  
- **Study time** has relatively smaller impact

DEPLOYMENT
A Streamlit web application was built to:
- Take user input (G1, G2, studytime, failures)  
- Predict final grade (G3)  
- Provide interpretation of results 

