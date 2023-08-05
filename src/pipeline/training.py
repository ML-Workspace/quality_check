import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.constant import MetaInformation
from src.utility import Utils
import os
from src.logs import logger
import joblib

class Training:

    def __init__(self, train_path, test_path):
        self.train_path= os.path.join(os.getcwd(), "artifacts/data","train.csv")
        self.test_path= os.path.join(os.getcwd(), "artifacts/data","test.csv")
        self.utils= Utils()

    def preprocess_data(self, train_path, test_path):

        train_data= self.utils.load_data(train_path)
        test_data= self.utils.load_data(test_path)


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
        self.utils.create_directory(MetaInformation.model_path)
        joblib.dump(model, os.path.join(MetaInformation.model_path, 'wime_model.pkl'))
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

if __name__ == "__main__":
    training= Training(MetaInformation.train_path, MetaInformation.test_path)
    training.main()
    
