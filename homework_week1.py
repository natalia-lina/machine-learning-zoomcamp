from warnings import simplefilter
import pandas as pd
import numpy as np

LAPTOPS = pd.read_csv("data/laptops.csv")

def pandas_version_Q1():
    '''Prints the installed Pandas version'''
    print(pd.__version__)

def records_count_Q2():
    '''Prints the number o records in the dataset'''
    print(len(LAPTOPS))

def laptop_counts_Q3():
    '''Prints the number of different brands of laptops in the dataset'''
    print(len(LAPTOPS["Brand"].value_counts()))

def missing_values_Q4():
    '''
    Prints the number of columns that
    contain missing values in the dataset
    '''
    i = 0
    for v in LAPTOPS.isnull().sum().values:
        if v != 0: # if v is not zero, the current column has missing values
            i += 1
    print(i)

def maximum_final_price_Q5():
    '''Prints the maximum final price of Dell notebooks in the dataset'''
    print(LAPTOPS[LAPTOPS["Brand"] == "Dell"]["Final Price"].max())

def median_value_of_screen_Q6():
    '''
    Prints whether the median of value in column \"Screen\" remains the same
    after executing the instructions:

    1.   Calculate the most frequent value of the same \"Screen\" column;
    2.   Use fillna method to fill the missing values in \"Screen\" column
         with the most frequent value from the previous step;
    '''
    prev = LAPTOPS["Screen"].median()
    LAPTOPS["Screen"].fillna(LAPTOPS["Screen"].mode()[0], inplace=True)
    print(prev != LAPTOPS["Screen"].median())

def sum_of_weights_Q7():
    '''
    Prints the sum of weights of a very specific and complicated sequence
    of instructions that result in an brute force linear regression.
    '''
    x = LAPTOPS[
        LAPTOPS["Brand"] == "Innjoo"][["RAM", "Storage", "Screen"]].to_numpy()
    y = [1100, 1300, 800, 900, 1000, 1100]
    print(
        np.sum(np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y))
    )


def main():
    simplefilter(action='ignore', category=FutureWarning)
    pandas_version_Q1()
    records_count_Q2()
    laptop_counts_Q3()
    missing_values_Q4()
    maximum_final_price_Q5()
    median_value_of_screen_Q6()
    sum_of_weights_Q7()

if __name__ == '__main__':
    main()
