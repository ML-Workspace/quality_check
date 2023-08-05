import os
import pandas as pd
class Utils:

    def __init__(self):
        pass

    def create_directory(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")


    def load_data(self, url):
        data = pd.read_csv(url, sep=',')
        return data