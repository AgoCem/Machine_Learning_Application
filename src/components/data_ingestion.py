import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

            # @dataclass is a decorator, that basically extend the DataIngestionConfig class 
            # functionalities with the methods contained in dataclass class
@dataclass 
class DataIngestionConfig:
    #inputs of the data ingestion component
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv') #reading the data from whatever source we have
            logging.info("Correctly read the dataset and saved it as a DataFrame")

            #creating a new directory (if it doesn't already exists) to store the data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            #storing the new data and also the train/test splitted data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split started")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=10)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path, #returning the data path to use it later on
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys) #using the custom exception we created to have everything more clear
        

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
        