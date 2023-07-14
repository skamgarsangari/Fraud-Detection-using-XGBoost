# Fraud-Detection-using-XGBoost

This Python script performs fraud detection using XGBoost, a popular machine-learning algorithm. It analyses transaction data and predicts whether a transaction is fraudulent or not.


## Installation

This script is compatible with Python 3.7 and above.

1 - Create and activate a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate # for Linux/Mac
env\Scripts\activate # for Windows


2- Install the required packages (already presented as a requirements.txt in the package directory):

pip install -r requirements.txt


## Usage

To run the script, first modify the working parameters and execute.

### Parameters

The script accepts the following parameters:

- --working_path (str): Working directory, contains the input files
- --transactions_fname (str): filename of csv file contains all transactions
- --labels_fname (str): filename of csv file detected fraud transaction
- --verbose (boolean): if True, return more reports on screen [Default: False]
- --threshold (float): Optimal threshold for fraud detection in our XGboost model [Default: 0.80]
- --test_size (float): fraction of data to be considered as test [Default: 0.0.3]
- --random_state (int): seed to generate the random number [Default: 2023]
- --MAX_ROUNDS (int): lgb iterations [Default: 200]
- --VERBOSE_EVAL (int): Print out metric result [Default: 10]


## Version
- Version: 0.0.1


## Author
- Author: Saeideh Kamgarsangari [saeideh.kamgar@gmail.com]

