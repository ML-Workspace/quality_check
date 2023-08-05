import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.utility import Utils
import os
from src.logs import logger
import joblib
import yaml
import argparse

class Training:

    def __init__(self, config_path):
        self.utils= Utils()
        self.config = self.utils.read_params(config_path)
        self.train_path= os.path.join(os.getcwd(), self.config['load_data']['train_path'])
        self.test_path= os.path.join(os.getcwd(), self.config['load_data']['test_path'])


    def preprocess_data(self, train_path, test_path):
        

        train_data= self.utils.load_data(train_path)
        test_data= self.utils.load_data(test_path)

        print(train_data.columns[0])

        y_train = train_data['quality']
        X_train = train_data.drop('quality', axis=1)
        
        y_test = test_data['quality']
        X_test = test_data.drop('quality', axis=1)
        

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    

    def train_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.utils.create_directory(self.config['model_dir'])
        joblib.dump(model, os.path.join(self.config['model_dir'], 'wime_model.pkl'))
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse


    def main(self):
        
        X_train_scaled, X_test_scaled, y_train, y_test = self.preprocess_data(self.train_path, self.test_path)
        
        model = self.train_model(X_train_scaled, y_train)
        
        mse = self.evaluate_model(model, X_test_scaled, y_test)
        print(f"Mean Squared Error: {mse}")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training= Training(config_path=parsed_args.config)
    training.main()
    
