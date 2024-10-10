import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "data/laptops.csv"
TARGET = "final_price"
FEATURES = ["ram","storage","screen"]

def snake_case_columns(columns: pd.Index):
    '''Normalizes all columns of a dataset into snake case format'''
    return columns.str.lower().str.replace(" ", "_")


def get_prepared_dataset():
    '''
    Returns dataframe with normalized column
    names and ignores columns that won"t be used
    '''
    df = pd.read_csv(DATA_PATH)
    df.columns = snake_case_columns(df.columns)
    return df[FEATURES+[TARGET]]

def get_shuffled_df(df, seed: int=42):
    '''
    Returns shuffled dataframe
    '''
    np.random.seed(seed)
    index = np.arange(len(df))
    np.random.shuffle(index)
    return df.iloc[index]


def get_train_val_test_split(df, distribution: tuple=(0.6, 0.2, 0.2)):
    '''
    Returns train, validation and test dataframes split
    from input dataframe, given the sets' distribution.
    '''
    len_df = len(df)
    n_train, n_val, n_test = tuple(map(lambda p: int(len_df*p), distribution))
    return df[:n_train], df[n_train:n_train+n_val], df[n_train+n_val:]


def fit_linear_regression(features, target, r=None):
    '''Fits linear regression given features and target variables'''
    feats_stack = np.column_stack(
        (np.ones(features.shape[0]), features)
    )
    trans = feats_stack.T.dot(feats_stack)
    if r is not None:
        trans = trans + r*np.eye(trans.shape[0])
    inv = np.linalg.inv(trans)
    weights = inv.dot(feats_stack.T).dot(target)

    return weights[0], weights[1:]

def predict_linear_regression(features, weights):
    return weights[0] + features.dot(weights[1])

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

def eda(df, column="final_price"):
    '''Plots histogram of values of a given column'''
    laptops[column].plot.hist()
    plt.show()

def missing_values_Q1(df):
    '''Prints column name of columns that have missing values'''
    for column in df.columns:
        if df[column].isnull().sum() != 0:
            print(column)

def median_Q2(df, column="ram"):
    '''Prints median of values from a given column'''
    print(df[column].median())

def filling_nas_Q3(train, val, column='screen'):
    train_zero = train.fillna(0)
    train_mean = train.fillna(train[column].mean())

    val_zero = val.fillna(0)
    val_mean = val.fillna(train[column].mean())

    weights_zero = fit_linear_regression(
        train_zero[FEATURES], train_zero[TARGET]
    )
    weights_mean = fit_linear_regression(
        train_mean[FEATURES], train_mean[TARGET]
    )

    pred_zero = predict_linear_regression(val_zero[FEATURES].values, weights_zero)
    pred_mean = predict_linear_regression(val_mean[FEATURES].values, weights_mean)

    rmse_zero = np.round(rmse(val_zero[TARGET].values, pred_zero), 2)
    rmse_mean = np.round(rmse(val_mean[TARGET].values, pred_mean),2)

    if rmse_zero > rmse_mean:
        print("Mean")
    elif rmse_zero < rmse_mean:
        print(0)
    else:
        print("Both")


def regularization_Q4(train, val, r_values=[0, 0.01, 0.1, 1, 5, 10, 100]):
    train = train.fillna(0)
    val = val.fillna(0)

    min_rmse = -1
    best_r = -1
    for r in r_values:
        weights = fit_linear_regression(
            train[FEATURES], train[TARGET], r
        )

        pred = predict_linear_regression(val[FEATURES].values, weights)
        score = np.round(rmse(val[TARGET].values, pred), 2)

        if score < min_rmse or min_rmse == -1:
            min_rmse = score
            best_r = r
    print(best_r)

def rmse_spread_Q5(df):
    seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scores = []
    for seed in seed_values:
        train, val, test = get_train_val_test_split(
            get_shuffled_df(df, seed=seed).fillna(0)
        )
        weights = fit_linear_regression(train[FEATURES], train[TARGET])
        pred = predict_linear_regression(val[FEATURES], weights)
        scores = scores + [np.round(rmse(pred, val[TARGET]), 3)]

    print(np.std(scores))


def rmse_on_test_Q5(df):
    train, val, test = get_train_val_test_split(
        get_shuffled_df(df, seed=9).fillna(0), distribution=(0.8, 0, 0.2)
    )

    weights = fit_linear_regression(train[FEATURES], train[TARGET],r=0.001)
    pred = predict_linear_regression(test[FEATURES].values, weights)
    score = np.round(rmse(test[TARGET].values, pred), 2)
    print(score)

def main():
    laptops = get_prepared_dataset()
    missing_values_Q1(laptops)
    median_Q2(laptops)
    train, val, test = get_train_val_test_split(
        get_shuffled_df(laptops)
    )
    filling_nas_Q3(train, val)
    regularization_Q4(train, val)
    rmse_spread_Q5(laptops)
    rmse_on_test_Q5(laptops)
    #print(train_linear_regression(train[FEATURES], train[TARGET]))
    #print(train[])


if __name__ == "__main__":
    main()
