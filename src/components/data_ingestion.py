import os
import sys
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation


# Initialize the Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Starts!")

        try:
            data = pd.read_csv(os.path.join('code/data', 'gemstone.csv'))
            logging.info("Data read as Pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            data.to_csv(self.ingestion_config.raw_data_path, index = False)
            logging.info("Train Test Split")

            train_set, test_set = train_test_split(data, test_size = 0.30)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e, sys)


# Run Data Ingestion

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)