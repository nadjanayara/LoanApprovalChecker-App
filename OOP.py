import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle as pkl
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
    
    def check_nulls(self):
        if self.data is not None:
            print("Null counts:\n", self.data.isnull().sum())

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        self.best_model = None
        self.encoders = {} 

    def CheckOutlierWithBoxPlot(self):
        numeric_columns = self.x_train.select_dtypes(include=['int64', 'float64']).columns
        self.x_train[numeric_columns].plot(kind='box', subplots=True, layout=(len(numeric_columns)//2+1, 2), figsize=(15,10), sharex=False)
        plt.tight_layout()
        plt.show()  

    def ReplaceGender(self, column):
        gender_replacements = {
            'fe male': 'female',
            'Male': 'male'
        }
        self.x_train[column] = self.x_train[column].replace(gender_replacements)
        self.x_test[column] = self.x_test[column].replace(gender_replacements)

    def filterAge(self, column='person_age', threshold=100):
        self.x_train = self.x_train[self.x_train[column] <= threshold]
        self.y_train = self.y_train[self.x_train.index]

        self.x_test = self.x_test[self.x_test[column] <= threshold]
        self.y_test = self.y_test[self.x_test.index]

        self.x_train[column] = self.x_train[column].astype(int)
        self.x_test[column] = self.x_test[column].astype(int)

    def ImputeNaNIncome(self, column):
        median_income = self.x_train[column].median()
        self.x_train[column] = self.x_train[column].fillna(median_income)
        self.x_test[column] = self.x_test[column].fillna(median_income)

    def ConvertToInt(self, column):
        self.x_train[column] = self.x_train[column].astype('int64')
        self.x_test[column] = self.x_test[column].astype('int64')

    def LabelEncode(self, column):
        encoder = LabelEncoder()
        self.x_train[column] = encoder.fit_transform(self.x_train[column])
        self.x_test[column] = encoder.transform(self.x_test[column])
        self.encoders[column] = encoder

    def OrdinalEncode(self, column, order_list):
        self.x_train[column] = pd.Categorical(self.x_train[column], categories=order_list, ordered=True).codes
        self.x_test[column] = pd.Categorical(self.x_test[column], categories=order_list, ordered=True).codes

    def OneHotEncode(self, column):
        self.x_train = pd.get_dummies(self.x_train, columns=[column], drop_first=False)
        self.x_test = pd.get_dummies(self.x_test, columns=[column], drop_first=False)

    def makePrediction(self):
        model_to_use = self.best_model if self.best_model is not None else self.model
        self.y_predict = model_to_use.predict(self.x_test)

    def createReport(self, y_test, y_predict):
        print('\nClassification Report\n')
        print(classification_report(y_test, y_predict, target_names=['0', '1']))

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def train_model(self, model=None):
        if model is None: 
            model = self.model
        model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def createModel(self, random_state=42):
        self.model = XGBClassifier(random_state=random_state)

    def HyperTuning(self):
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 10]
        }

        xgb = XGBClassifier()
        xgb_search = GridSearchCV(xgb,
                                  param_grid=param_grid,
                                  scoring='accuracy',
                                  cv=5, verbose=1, n_jobs=-1)

        xgb_search.fit(self.x_train, self.y_train)
        print("Best Hyperparameters :", xgb_search.best_params_)
        self.best_model = xgb_search.best_estimator_

    def save_model_to_file(self, filename, model):
        with open(filename, 'wb') as file:
            pkl.dump(model, file)

    def save_encoders(self):
        for name, encoder in self.encoders.items():
            with open(f"{name}_encoder.pkl", 'wb') as file:
                pkl.dump(encoder, file)

file_path = "Dataset_A_loan.csv"
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.check_nulls()
data_handler.create_input_output('loan_status')

model_handler = ModelHandler(data_handler.input_df, data_handler.output_df)
model_handler.split_data()
model_handler.ReplaceGender(column='person_gender')
model_handler.filterAge('person_age', 100)
model_handler.ImputeNaNIncome('person_income')
model_handler.ConvertToInt(['cb_person_cred_hist_length'])

model_handler.LabelEncode('person_gender')
model_handler.LabelEncode('previous_loan_defaults_on_file')

education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
model_handler.OrdinalEncode('person_education', education_order)
model_handler.OneHotEncode('person_home_ownership')
model_handler.OneHotEncode('loan_intent')

model_handler.HyperTuning()
model_handler.train_model(model_handler.best_model)
model_handler.makePrediction()
model_handler.createReport(model_handler.y_test, model_handler.y_predict)

model_handler.save_model_to_file("best_xgbmodel.pkl", model_handler.best_model)
model_handler.save_encoders()
