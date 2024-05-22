import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_accuracies = []

        pass

    def predict(self,modelname,features):
        try:
            print("here")
            data_scaled = np.array(features)

            if modelname == "Linear Regression":
                model_path = os.path.join("Models","lr_model.pkl")
                model=load_object(file_path=model_path)
            elif modelname == "Ridge":
                model_path = os.path.join("Models","r_model.pkl")
                model=load_object(file_path=model_path)
            elif modelname == "Decision Tree":
               model_path = os.path.join("Models","dec_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname == "AdaBoost Regressor":
               model_path = os.path.join("Models","ada_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname == "Gradient Boosting":
               model_path = os.path.join("Models","gra_model.pkl")
               model=load_object(file_path=model_path)
               check_is_fitted(model)
            elif modelname == "Random Forest":
               model_path = os.path.join("Models","rnd_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname == "XGBRegressor":
               model_path = os.path.join("Models","xgb_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname=="CatBoosting Regressor":
                model_path = os.path.join("Models","cat_model.pkl")
                model=load_object(file_path=model_path)
            else:
                model_path = os.path.join("Models","lasso_model.pkl")
                model=load_object(file_path=model_path)
        

            # model_path = os.path.join("Modelsts","model.pkl")
            preprocessor_path=os.path.join('preprocessor','preprocessor.pkl')
            print("Before Loading")
            # model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = pd.DataFrame(data_scaled, columns=features.columns)           
            data_scaled=preprocessor.transform(data_scaled)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
    def accuracy(self,train_array,test_array,modelname):
        try:
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            if modelname == 'Linear Regression':
            # model = lr_model1
                model_path = os.path.join("Models","lr_model.pkl")
                model=load_object(file_path=model_path)
            elif modelname == "Ridge":
                # model = r_model1
                model_path = os.path.join("Models","r_model.pkl")
                model=load_object(file_path=model_path)
            elif modelname == "Decision Tree":
                # model = dec
               model_path = os.path.join("Models","dec_model.pkl")
               model=load_object(file_path=model_path)

            elif modelname == "AdaBoost Regressor":
                # model = dec
               model_path = os.path.join("Models","ada_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname == "Gradient Boosting":
                # model = dec
               model_path = os.path.join("Models","gra_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname == "Random Forest":
                # model = dec
               model_path = os.path.join("Models","rnd_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname == "XGBRegressor":
                # model = dec
               model_path = os.path.join("Models","xgb_model.pkl")
               model=load_object(file_path=model_path)
            elif modelname=="CatBoosting Regressor":
                # model=xgb
                model_path = os.path.join("Models","cat_model.pkl")
                model=load_object(file_path=model_path)
            else:
                model_path = os.path.join("Models","lasso_model.pkl")
                model=load_object(file_path=model_path)
        

            # model_path = os.path.join("Modelsts","model.pkl")
            model_accuracy = model.score(X_train, y_train)
            # model_accuracies = self.model_accuracies.append({'Model_Name': modelname, 'Accuracy': model_accuracy}, ignore_index=True)
            return model_accuracy
        except Exception as e:
            raise CustomException(e,sys)
           
    def calculate_accuracy(self, predictions, target):
        # Calculate the accuracy as the percentage of correct predictions
        accuracy = (predictions == target).mean() * 100

        return accuracy
    
class CustomData:
    def __init__(  self,
        gender: str,
        university_type: str,
        extra_curricular: str,
        attendance: str,
        test_preparation_course: str,
        test_score: int,
        prep_score: int,
        name:str):

        self.gender = gender
        self.university_type = university_type
        self.extra_curricular = extra_curricular
        self.attendance = attendance
        self.test_preparation_course = test_preparation_course
        self.test_score = test_score
        self.prep_score = prep_score
        self.name = name

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "university_type": [self.university_type],
                "extra_curricular": [self.extra_curricular],
                "attendance": [self.attendance],
                "test_preparation_course": [self.test_preparation_course],
                "test_score": [self.test_score],
                "prep_score": [self.prep_score],
                "name" :[self.name]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

