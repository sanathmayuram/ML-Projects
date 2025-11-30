import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustonException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
        
    except Exception as e:
        raise CustonException(e, sys)
    
def evaluate_models(X_train, Y_train, X_test, Y_test, models, params):
    try:
        report = {}

        # Loop using model names instead of values
        for model_name, model in models.items():

            # Get the parameter grid using the correct key
            param_grid = params.get(model_name, {})

            # Hyperparameter tuning
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, Y_train)

            # Set best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_score = r2_score(Y_train, y_train_pred)
            test_score = r2_score(Y_test, y_test_pred)

            # Save the final test score
            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustonException(e, sys)
    

def load_object(file_path):
    try :
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustonException(e, sys)

