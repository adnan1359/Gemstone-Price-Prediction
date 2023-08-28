import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            # Numerical Pipeline

            logging.info("Pipeline Initiated")

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories = [
                        cut_categories, color_categories, clarity_categories])),
                        ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            logging.info("Pipeline Completed")
            return preprocessor


        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Train and Test data read successfully")
            logging.info(f'Train DataFrame Head: \n{train_data.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n{test_data.head().to_string()}')

            logging.info('Obtaining Preprocessing Object')

            preprocesing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'Unnamed: 0']

            input_feature_train_data = train_data.drop(columns = drop_columns, axis = 1)
            target_feature_train_data = train_data[target_column_name]

            input_feature_test_data = test_data.drop(columns = drop_columns, axis = 1)
            target_feature_test_data = test_data[target_column_name]

            # Transforming using preprocessing object
            input_feature_train_arr = preprocesing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocesing_obj.transform(input_feature_test_data)

            logging.info("Applying preprocessing object on training and testing dataset")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocesing_obj
            )

            logging.info("Preprocessor File Saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
 
        except Exception as e:
            logging.info("Exception occured at the initiate data tranformation stage")
            raise CustomException(e, sys)