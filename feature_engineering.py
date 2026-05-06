import pandas as pd

def feature_engineering(df):

    df["price_band"] = pd.cut(df["Price"],
                              bins=[0,200,400,1000],
                              labels=["low","mid","high"])

    df["exp_level"] = pd.cut(df["Experience"],
                             bins=[0,3,7,20],
                             labels=["junior","mid","senior"])

    df["rating_level"] = pd.cut(df["Rating"],
                                bins=[0,3,4.5,5],
                                labels=["low","good","excellent"])

    df = pd.get_dummies(df, drop_first=True)

    return df