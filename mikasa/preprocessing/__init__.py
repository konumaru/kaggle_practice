import numpy as np
import pandas as pd


def add_dummies(data: pd.DataFrame, column: str, drop: bool = True):
    ohe = pd.get_dummies(data[column]).add_prefix(f"{column}_")
    data = data.drop(column, axis=1)
    data = data.join(ohe)
    return data
