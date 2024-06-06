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
            # this is only an example

            params={
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Random_Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient_Boost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear_Reg":{},
                "Lasso_Reg" : {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'max_iter': [1000, 5000, 10000],
                    'selection': ['cyclic', 'random']
                },
                "Ridge_Reg" : {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    'max_iter': [1000, 5000, 10000]
                },
                "Elastic_Net" : {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [1000, 5000, 10000]
                },
                "SVR" : {
                    'kernel': ['linear', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.1, 0.2, 0.5, 1.0]
                },
                "KNeighbor" : {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [10, 30, 50],
                    'p': [1, 2]
                },
                "Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
            }

            model_report:dict=evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test,y_test = y_test, models = models, params = params)

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
