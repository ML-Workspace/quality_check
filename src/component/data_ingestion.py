import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from src.constant import MetaInformation
import os
from src.utility import Utils
from src.logs import logger 


class DataIngestion:

    def __init__(self, url):
        self.url= url
        self.utils= Utils()


    def save_csv_to_directory(self, data_frame, directory_path, file_name):
        file_path = os.path.join(directory_path, file_name)
        data_frame.to_csv(file_path, index=False)
        print(f"CSV file '{file_name}' saved to '{directory_path}'.")

    def _init_data_ingetion(self):

        df= self.utils.load_data(self.url)
        df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.93*len(df)), int(.99*len(df))])
        logger.info(df_train.shape)
        logger.info(df_test.shape)
        logger.info(df_val.shape)

        self.utils.create_directory(MetaInformation.directory_path)

        self.save_csv_to_directory(df, MetaInformation.directory_path, MetaInformation.raw_data_path)
        self.save_csv_to_directory(df_train, MetaInformation.directory_path, MetaInformation.train_path)
        self.save_csv_to_directory(df_test, MetaInformation.directory_path, MetaInformation.test_path)
        self.save_csv_to_directory(df_val, MetaInformation.directory_path, MetaInformation.val_path)


if __name__== '__main__':
   ingestion= DataIngestion(MetaInformation.raw_data_url)
   ingestion._init_data_ingetion()