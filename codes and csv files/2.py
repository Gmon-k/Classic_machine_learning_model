"""
Organize your data
Divide your data into a training set and a test set. Write a program that takes in the name of the data set
CSV file and a parameter that specifies the percent of the data to use as a test set. Ideally this information 
should come from command-line parameters. The program should generate two CSV files:

Submitted by : Gmon Kuzhiyanikkal
"""


import csv
import random
import sys

#function to for data splitting to test and train set
def data_spliter(csv_file_name, percentage_test):
    with open(csv_file_name, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    random.shuffle(data)

    #splitting the data depending the percentage prvovided in the command line.
    test_size = int(len(data) * percentage_test)
    test_set = data[:test_size]
    train_set = data[test_size:]

    #name of the test file and train file generated
    test_file = f'test_set.csv'
    train_file = f'train_set.csv'

    #writing the data to test set file
    with open(test_file, 'w') as test_file:
        writer = csv.writer(test_file)
        writer.writerows(test_set)


    #writing the data to train set file
    with open(train_file, 'w') as train_file:
        writer = csv.writer(train_file)
        writer.writerows(train_set)


    
    #priting the output
    print("\n")
    print(f'Test set percentage: {percentage_test * 100}% ({len(test_set)} samples)')
    print("\n")
    print(f'Train set percentage: {(1 - percentage_test) * 100}% ({len(train_set)} samples)')
    print("\n")
    print(f'Test file name: test_set.csv')
    print("\n")
    print(f'Train file name: train_set.csv')
    print('\n')

#main function for taking the command line argument
if __name__ == "__main__":
    file_name = sys.argv[1]
    test_percentage = float(sys.argv[2])
    data_spliter(file_name, test_percentage)
