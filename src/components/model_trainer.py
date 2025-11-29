import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustonException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class MOdelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = MOdelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting Train and Testinput Data")
            X_train, Y_train , X_test, Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decsion Tree" :
                DecisionTreeRegressor(),
                "Gradiant Boosting" :
                GradientBoostingRegressor(),
                "Liner Regression" :
                LinearRegression(),
                "K-Neighbors Classifier" :
                KNeighborsRegressor(),
                "XG Boost Classifier" :
                XGBRegressor(),
                "CatBoosting Classifier" :
                CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" :
                AdaBoostRegressor()
            }

            model_report:dict=evaluate_models(X_train=X_train,Y_train = Y_train,X_test = X_test, Y_test= Y_test , models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustonException("No best model found")
            
            logging.info(f"Best Found model on the both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(Y_test, predicted)
            return r2
        

        except Exception as e:
            raise CustonException(e, sys)
            
