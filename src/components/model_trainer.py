import sys
import os
from data_transformation import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1], #here i take the data based on how i constructed the train and test
                test_array[:,:-1], #arrays in the datatransformation.py
                test_array[:,-1]
            )

            models = {
                "CatBoost" : CatBoostRegressor(),
                "XGBoost" : XGBRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "Random_Forest" : RandomForestRegressor(),
                "Gradient_Boost" : GradientBoostingRegressor(),
                "Linear_Reg" : LinearRegression(),
                "Lasso_Reg" : Lasso(),
                "Ridge_Reg" : Ridge(),
                "Elastic_Net" : ElasticNet(),
                "SVR" : SVR(),
                "KNeighbor" : KNeighborsRegressor(),
                "Tree" : DecisionTreeRegressor()
            } #here i use every model that i imported, the dataset is small and
            # this is only an example, i will not hyperparameter tuning and i only will
            # use the default parameters

            model_report:dict=evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test,y_test = y_test, models = models)

            ## to get best model score from the report
            best_model_score = max(sorted(model_report.values()))

            ## to get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No proper model found")
            
            logging.info(f"Best found model on both training and testing dataset which is {best_model_name} with an accuracy of {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_squared = r2_score(y_test,predicted)

            return r2_squared

        except Exception as e:
            raise CustomException(e,sys)
