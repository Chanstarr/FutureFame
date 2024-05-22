import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    bestmodel_path = os.path.join("Models", "model.pkl")
    path1=os.path.join("Models", "lr_model.pkl")
    path2=os.path.join("Models", "r_model.pkl")
    path3=os.path.join("Models", "dec_model.pkl")
    path4=os.path.join("Models", "ada_model.pkl")
    path5=os.path.join("Models", "gra_model.pkl")
    path6=os.path.join("Models", "rnd_model.pkl")
    path7=os.path.join("Models", "xgb_model.pkl")
    path8=os.path.join("Models", "cat_model.pkl")
    path9=os.path.join("Models", "lasso_model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "Ridge":Ridge(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(),
                "Lasso":Lasso()
            }
            # Best way is to use additional config file, yaml file and from that can read hyperparameters
            params={
                "Decision Tree": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # # 'splitter':['best','random'],
                    # # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Ridge":{},
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }        
        
            # model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
            #                                  models=models,param=params)
            
            # # to get best model score from dict
            # best_model_score = max(sorted(model_report.values()))

            # # best model score name
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # best_model = models[best_model_name]
            model1=models["Linear Regression"]
            model2=models["Ridge"]
            model3=models["Decision Tree"]
            model4=models["AdaBoost Regressor"]
            model5=models["Gradient Boosting"]
            model6=models["Random Forest"]
            model7=models["XGBRegressor"]
            model8=models["CatBoost Regressor"]
            model9=models["Lasso"]
            
            model1.fit(X_train, y_train)
            model2.fit(X_train, y_train)
            model3.fit(X_train, y_train)
            model4.fit(X_train, y_train)
            model5.fit(X_train, y_train)
            model6.fit(X_train, y_train)
            model7.fit(X_train, y_train)
            model9.fit(X_train,y_train)
            # model8.fit(X_train, y_train)
            # best_model.fit(X_train,y_train)
            # best_model_score=model_report.get('best_model')
            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            logging.info(f"Models on training and testing dataset")

            # save_object(
            #     file_path=self.model_trainer_config.bestmodel_path,
            #     obj=best_model
            # )
            save_object(file_path=self.model_trainer_config.path1,obj=model1)
            save_object(file_path=self.model_trainer_config.path2,obj=model2)
            save_object(file_path=self.model_trainer_config.path3,obj=model3)
            save_object(file_path=self.model_trainer_config.path4,obj=model4)
            save_object(file_path=self.model_trainer_config.path5,obj=model5)
            save_object(file_path=self.model_trainer_config.path6,obj=model6)
            save_object(file_path=self.model_trainer_config.path7,obj=model7)
            save_object(file_path=self.model_trainer_config.path8,obj=model8)
            save_object(file_path=self.model_trainer_config.path9,obj=model9)

            # predicted = model1.predict(X_test)

            # r2_square = r2_score(y_test,predicted)
            # return r2_square
        except Exception as e:
            raise CustomException(e,sys)
