import os 
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception  import CustomException
from src.logger import logging 


from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'XGBoost Regressor': XGBRegressor(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }
            
            
            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models=models)
            
            ## To get best model score from dict 
            best_model_score  = max(sorted(model_report.values()))
            
            ## To get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
             
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            
            
            logging.info(f'Best model found  on both training  and testing dataset')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model 
            )
            
            predicated = best_model.predict(x_test)
            
            r2_square  = r2_score(y_test,predicated)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)

