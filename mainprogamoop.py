import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import LabelEncoder


class DataHandler: #untuk bagi input data dan output data
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)
    
    def fixGenderColumnInconsistency(self):
        self.input_df['person_gender'] = self.input_df['person_gender'].replace({'Male':'male', 'fe male': 'female'})

# ModelHandler Class
class ModelHandler: # preprocessing data
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def checkOutlierWithBox(self, kolom):
        boxplot = self.x_train.boxplot(column=[kolom]) 
        plt.show()
        
    def createMedianFromColumn(self, kolom):
        return np.median(self.x_train[kolom])
    
    def createModel(self, eta=0.5, gamma=0, max_depth=3):
         self.model = xgb.XGBClassifier(eta=eta, gamma=gamma, max_depth=max_depth)
    
    def fillingNAWithNumbers(self, columns, number):
        self.x_train[columns] = self.x_train[columns].fillna(number)
        self.x_test[columns] = self.x_test[columns].fillna(number)

    def lowerUpperBound(self, column):
        q1 = np.nanpercentile(self.x_train[column], 25)
        q3 = np.nanpercentile(self.x_train[column], 75)
        IQR = q3-q1
        lower_bound = q1-(1.5*IQR)
        upper_bound = q3+(1.5*IQR)
        return lower_bound, upper_bound

    def replaceOutlierWithMedian(self, column, median, lower_bound, upper_bound):
        self.x_train[column] = self.x_train[column].apply(
            lambda x: median if (x < lower_bound or x > upper_bound) else x
        )
        self.x_test[column] = self.x_test[column].apply(
            lambda x: median if (x < lower_bound or x > upper_bound) else x
        )

    def encodeCategoricalVariables(self):
        le_gender = LabelEncoder()
        le_edu = LabelEncoder()
        le_hownership = LabelEncoder()
        le_intent = LabelEncoder()
        le_prev_loan = LabelEncoder()

        self.x_train['person_gender'] = le_gender.fit_transform(self.x_train['person_gender'])
        self.x_train['person_education'] = le_edu.fit_transform(self.x_train['person_education'])
        self.x_train['person_home_ownership'] = le_hownership.fit_transform(self.x_train['person_home_ownership'])
        self.x_train['loan_intent'] = le_intent.fit_transform(self.x_train['loan_intent'])
        self.x_train['previous_loan_defaults_on_file'] = le_prev_loan.fit_transform(self.x_train['previous_loan_defaults_on_file'])

        self.x_test['person_gender'] = le_gender.transform(self.x_test['person_gender'])
        self.x_test['person_education'] = le_edu.transform(self.x_test['person_education'])
        self.x_test['person_home_ownership'] = le_hownership.transform(self.x_test['person_home_ownership'])
        self.x_test['loan_intent'] = le_intent.transform(self.x_test['loan_intent'])
        self.x_test['previous_loan_defaults_on_file'] = le_prev_loan.transform(self.x_test['previous_loan_defaults_on_file'])

        # get label mapping
        print(dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))
        print(dict(zip(le_edu.classes_, le_edu.transform(le_edu.classes_))))
        print(dict(zip(le_hownership.classes_, le_hownership.transform(le_hownership.classes_))))
        print(dict(zip(le_intent.classes_, le_intent.transform(le_intent.classes_))))
        print(dict(zip(le_prev_loan.classes_, le_prev_loan.transform(le_prev_loan.classes_))))
    
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))
            
    def split_data(self, test_size=0.2, random_state=88):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)
    
    def tuningParameter(self):
        parameters = {
            'eta': [0.2, 0.3, 0.5],
            'gamma': [0, 0.1, 0.3, 0.5],
            'max_depth': [3,5,7]
        }
        XGB_model = xgb.XGBClassifier()
        XGB_model= GridSearchCV(XGB_model ,
                            param_grid = parameters,   # hyperparameters
                            scoring='accuracy',        # metric for scoring
                            cv=5)
        XGB_model.fit(self.x_train,self.y_train)
        print("Tuned Hyperparameters :", XGB_model.best_params_)
        print("Accuracy :",XGB_model.best_score_)
        self.createModel(eta =XGB_model.best_params_['eta'], gamma=XGB_model.best_params_['gamma'], max_depth=XGB_model.best_params_['max_depth'])

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  # Open the file in write-binary mode
            pickle.dump(self.model, file)  # Use pickle to write the model to the file



file_path = 'Dataset_A_loan.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('loan_status')
data_handler.fixGenderColumnInconsistency()
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

# cek outlier
#model_handler.checkOutlierWithBox('person_age')
#model_handler.checkOutlierWithBox('person_income')
#model_handler.checkOutlierWithBox('person_emp_exp')
#model_handler.checkOutlierWithBox('loan_amnt')
#model_handler.checkOutlierWithBox('loan_int_rate')
#model_handler.checkOutlierWithBox('loan_percent_income')
#model_handler.checkOutlierWithBox('cb_person_cred_hist_length')
#model_handler.checkOutlierWithBox('credit_score')

model_handler.replaceOutlierWithMedian(
    'person_age',
    model_handler.createMedianFromColumn('person_age'),
    *model_handler.lowerUpperBound('person_age')
)

model_handler.replaceOutlierWithMedian(
    'person_income',
    model_handler.createMedianFromColumn('person_income'),
    *model_handler.lowerUpperBound('person_income')
)

model_handler.replaceOutlierWithMedian(
    'person_emp_exp',
    model_handler.createMedianFromColumn('person_emp_exp'),
    *model_handler.lowerUpperBound('person_emp_exp')
)

model_handler.replaceOutlierWithMedian(
    'loan_amnt',
    model_handler.createMedianFromColumn('loan_amnt'),
    *model_handler.lowerUpperBound('loan_amnt')
)

model_handler.replaceOutlierWithMedian(
    'loan_int_rate',
    model_handler.createMedianFromColumn('loan_int_rate'),
    *model_handler.lowerUpperBound('loan_int_rate')
)

model_handler.replaceOutlierWithMedian(
    'loan_percent_income',
    model_handler.createMedianFromColumn('loan_percent_income'),
    *model_handler.lowerUpperBound('loan_percent_income')
)

model_handler.replaceOutlierWithMedian(
    'cb_person_cred_hist_length',
    model_handler.createMedianFromColumn('cb_person_cred_hist_length'),
    *model_handler.lowerUpperBound('cb_person_cred_hist_length')
)

model_handler.replaceOutlierWithMedian(
    'credit_score',
    model_handler.createMedianFromColumn('credit_score'),
    *model_handler.lowerUpperBound('credit_score')
)

#impute missing value
income_replace_na = model_handler.createMedianFromColumn('person_income')
model_handler.fillingNAWithNumbers('person_income',income_replace_na)

model_handler.encodeCategoricalVariables()

print("Before Tuning Parameter")
model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()
print("After Tuning Parameter")
model_handler.tuningParameter()
model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()
model_handler.save_model_to_file('trained_model.pkl') 



