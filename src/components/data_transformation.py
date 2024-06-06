import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        #We will always call this function when we want to perform the data transformation
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ] #here we are dividing the columns in order to handle them easily after

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")), #with the imputer we handle missing data
                                # in this case we use the median in order to not be influenced by outliers
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")), #since it is a categorical variable
                                                #we try to use the most frequent value as imputer
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)) #In this case we only have a one hot encoder, so the
                                #so it is useless to scale, but it could be useful if we use some other encoder
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            # now we will use the column transformer to apply the several preprocessing and transformation
            # pipelines that we defined

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data succesfully read")

            logging.info("Obtaining preprocessing objects")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 'columns')
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 'columns')
            target_feature_test_df = test_df[target_column_name]

            logging.info("Starting the preprocessing on training and test data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)#of course
                        #the processor is fit only on the train data and then i apply that fit
                        # on test data as well, that's why it is only transform


            #a useful numpy feature to concatenate over columns
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            #a useful numpy feature to concatenate over columns
            test_arr = np.c_[ 
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("Correctly saved the preprocessor")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

