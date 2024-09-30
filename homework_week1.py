import pandas as pd
import numpy as np

LAPTOPS = pd.read_csv("laptops.csv")

def pandas_version_Q1():
    print(pd.__version__)

def records_count_Q2():
    print(len(LAPTOPS))

def laptop_counts_Q3():
    print(len(LAPTOPS["Brand"].value_counts()))

def missing_values_Q4():
    i = 0
    for v in LAPTOPS.isnull().sum().values:
        if v != 0:
            i += 1
    print(i)

def maximun_final_price_Q5():
    print(LAPTOPS[LAPTOPS["Brand"] == "Dell"]["Final Price"].max())

def median_value_of_screen_Q6():
    prev = LAPTOPS["Screen"].median()
    LAPTOPS["Screen"].fillna(LAPTOPS["Screen"].mode()[0], inplace=True)
    print(prev != LAPTOPS["Screen"].median())

def sum_of_weights_Q7():
    x = LAPTOPS[LAPTOPS["Brand"] == "Innjoo"][["RAM", "Storage", "Screen"]].to_numpy()
    y = [1100, 1300, 800, 900, 1000, 1100]
    print(np.sum(np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)))


pandas_version_Q1()
records_count_Q2()
laptop_counts_Q3()
missing_values_Q4()
maximun_final_price_Q5()
median_value_of_screen_Q6()
sum_of_weights_Q7()
