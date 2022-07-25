from audioop import rms
import os
from random import random
import mlflow
import argparse
import time
from pandas import PeriodDtype
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd

def evaluate(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mse = mean_squared_error(actual, pred)
    r2 = r2_score(actual,pred)

    return rmse, mse, r2

def get_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL, sep = ";")
        return df
    except Exception as e:
        raise e


def main(alpha,l1_ratio):
    target = "quality"
    df = get_data()
    train, test = train_test_split(df)

    train_y = train["quality"]
    train_x = train.drop(columns = ["quality"])

    test_y = test["quality"]
    test_x = test.drop(columns = ["quality"])
    
    model_lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,random_state=42)
    model_lr.fit(X=train_x,y=train_y)

    pred = model_lr.predict(test_x)

    rmse, mse, mae = evaluate(actual=test_y, pred=pred)

    print(f"params : aplha ={alpha}, l_1 ration= {l1_ratio}")
    print(f"metrics :   mse={mse}, rmse = {rmse}, mae = {mae}")

    with mlflow.start_run():
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        metric =[rmse, mse, mae]
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("mse",mse)

        os.makedirs("temp", exist_ok=True)
        with open("temp/sample.txt","w") as f:
            f.write(time.asctime())
        mlflow.log_artifacts("temp")
        mlflow.sklearn.log_model(model_lr,"model" )

if __name__ =="__main__":
    args= argparse.ArgumentParser()
    args.add_argument("--alpha","-a",type = float, default=0.5)
    args.add_argument("--l1_ratio","-l1",type = float, default=0.5)
    parsed_args = args.parse_args()

    main(parsed_args.alpha, parsed_args.l1_ratio)


