"""
Fraud Detection using XGBoost

Description:
This script trains an XGBoost classifier to detect fraudulent transactions.

Author: Saeideh Kamgar [saeideh.kamgar@gmail.com]
Date: 28 May 2023
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import category_encoders as ce
import matplotlib.cm as cm
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# ----------------------------------------------------------------------------------------

def fraud_XGboost(
    working_path,
    transactions_fname,
    labels_fname,
    verbose=False,
    threshold = 0.80,
    test_size=0.3,
    random_state=2023,
    MAX_ROUNDS=200,
    VERBOSE_EVAL=10,
):
    """
    Purpose:
    preprocesing of a set of transactions and trains an 
    XGBoost classifier to detect fraudulent transactions
    
    Args::
        working_path (str): Working directory, contains the input files
        transactions_fname (str): filename of csv file contains all transactions
        labels_fname (str): filename of csv file detected fraud transaction
        verbose (boolean): if True, return more reports on screen [Default: False]
        threshold (float): Optimal threshold for fraud detection in our XGboost model [Default: 0.80]
        test_size (float): fraction of data to be considered as test [Default: 0.0.3]
        random_state (int): seed to generate the random number [Default: 2023]
        MAX_ROUNDS (int): lgb iterations [Default: 200]
        VERBOSE_EVAL (int): Print out metric result [Default: 10]

    Example:
    Modify the working parameters and run the main script

    """

    # Taking care of inputs/outputs
    # ------------------------------------------------------------------------------------

    # check if working_path exists
    assert Path(working_path).is_dir(), f'No {working_path} directory exists!'

    # Define paths for data, models, and figures
    data_path = Path(working_path) / "data"
    Path(data_path).mkdir(parents=True, exist_ok=True)

    model_path = Path(working_path) / "model"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    fig_path = Path(working_path) / "figures"
    Path(fig_path).mkdir(parents=True, exist_ok=True)



    # Loading data into pandas DF
    trans_file = data_path / transactions_fname
    assert Path(trans_file).is_file(), f'No transactions file named {transactions_fname} found in {data_path}'
    labels_file = data_path / labels_fname
    assert Path(labels_file).is_file(), f'No labels file named {labels_fname} found in {data_path}'
    trans_data = pd.read_csv(trans_file)
    label_data = pd.read_csv(labels_file)


    if verbose:
        # Display trans dataset information
        print(trans_data.describe())
        print(trans_data.info())
        print(trans_data.shape)
        print(trans_data.head(20))

        # Display label dataset information
        print(label_data.describe())
        print(label_data.info())
        print(label_data.shape)
        print(label_data.head(20))

    # Data pre-processing: [part I]
    # ------------------------------------------------------------------------------------

    # Convert categorical features to object data type.
    trans_data["merchantCountry"] = trans_data["merchantCountry"].astype("object")
    trans_data["mcc"] = trans_data["mcc"].astype("object")
    trans_data["posEntryMode"] = trans_data["posEntryMode"].astype("object")

    # Merge trans_data and label_data based on a common column.
    # Cross-match 'fraud' column using the common column.
    # Define the common column used for cross-matching.
    common_col = "eventId"

    # Add a column of zeros to the trans_data dataset.
    trans_data["fraud"] = 0

    # Cross-match 'fraud' column in both datasets using the common column.
    fraud_loc = trans_data[common_col].isin(label_data[common_col])
    trans_data.loc[fraud_loc, "fraud"] = 1

    # Merge the 'reportedTime' column from label data to trans data based on the common column.
    merged_df = pd.merge(trans_data, label_data, on="eventId", how="outer")
    # Set reportedTime to 0 if fraud is 0, otherwise copy the reportedTime value.
    merged_df["reportedTime"] = np.where(
        merged_df["fraud"] == 1, merged_df["reportedTime"], 0
    )

    # Check if the data is imbalanced.
    label_freq = merged_df.fraud.value_counts()
    if verbose:
        print("Label by type:\n", label_freq)

    # Calculate the fraudulent rate and print a warning message if it is below 0.3 (30%).
    isfraud_rate = abs(label_freq[1] / label_freq.sum())
    if isfraud_rate < 0.3:
        print(
            "WARNING: Data is imbalanced with a",
            round(isfraud_rate, 3),
            "fraudulent rate",
        )

    # There are missing values in colum 'merchantZip' in dataset
    # No pattern is detected in dataset for missing values.
    # Replace missing values in the 'merchantZip' column with 'unknown'. This is a new category for Zip code. 
    merged_df["merchantZip"] = merged_df["merchantZip"].fillna("unknown")

    # Data pre-processing: [part II]
    # ------------------------------------------------------------------------------------

    # Define a list of non-categorical features and label.
    isnot_categorical_columns = ["transactionAmount", "availableCash", "fraud"]
    
    ## Note: A Log-transform of 'transactionAmount' do not improve the model.   

    # Set the plot style and size.
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 7))

    # Determine the number of features and the maximum number of subplots (2).
    num_features = len(isnot_categorical_columns)
    num_subplots = min(num_features, 2)

    # Create subplots for each feature and plot the KDE distributions for fraud = 0 and fraud = 1.
    for i, feature in enumerate(isnot_categorical_columns[:num_subplots], start=1):
        plt.subplot(1, num_subplots, i)
        sns.kdeplot(
            merged_df.loc[merged_df["fraud"] == 0][feature],
            bw_method=0.5,
            label="Fraud = 0",
        )
        sns.kdeplot(
            merged_df.loc[merged_df["fraud"] == 1][feature],
            bw_method=0.5,
            label="Fraud = 1",
        )
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis="both", which="major", labelsize=12)

    # Adjust the layout, save the figure, and close the plot.
    plt.tight_layout()
    figure_name = "figure1.png"
    plt.savefig(str(fig_path / figure_name))
    print(f'LOG: Figure: {figure_name} generated in {fig_path}')
    plt.close()

    # Data pre-processing: [part III]
    # Extract time-date related features from the transaction Time
    # ------------------------------------------------------------------------------------

    # Ensure columns with date/time information have the correct datetime type
    trans_time_col = "transactionTime"
    report_time_col = "reportedTime"

    merged_df[trans_time_col] = pd.to_datetime(merged_df[trans_time_col])
    merged_df[report_time_col] = pd.to_datetime(merged_df[report_time_col])

    # Decompose time by extracting month and hour from datetime columns
    merged_df[f"{trans_time_col}_month"] = merged_df[trans_time_col].dt.month
    merged_df[f"{trans_time_col}_hour"] = merged_df[trans_time_col].dt.hour

    # Time patterns
    # Set the figure parameters and path
    bins = 50
    num_hours = 24
    num_months = 12

    # Extract hour distributions for fraud and valid transactions
    valid_hours = merged_df.loc[merged_df["fraud"] == 0][f"{trans_time_col}_hour"]
    fraud_hours = merged_df.loc[merged_df["fraud"] == 1][f"{trans_time_col}_hour"]
    hist_data = [valid_hours, fraud_hours]

    # Plot the histograms for fraud and valid transactions by hour
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    fraud_hours.hist(bins=num_hours, color="red")
    plt.title("Fraud transactions by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("# of transactions")
    plt.subplot(1, 2, 2)
    valid_hours.hist(bins=num_hours, color="green")
    plt.title("Valid transactions by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("# of transactions")
    plt.tight_layout()
    figure_name = "figure2.png"
    plt.savefig(str(fig_path / figure_name))
    print(f'LOG: Figure: {figure_name} generated in {fig_path}')
    plt.close()

    # Extract month distributions for fraud and valid transactions
    valid_months = merged_df.loc[merged_df["fraud"] == 0][f"{trans_time_col}_month"]
    fraud_months = merged_df.loc[merged_df["fraud"] == 1][f"{trans_time_col}_month"]

    # Plot the histograms for fraud and valid transactions by month
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    fraud_months.hist(bins=num_months, color="red")
    plt.title("Fraud transactions by Month")
    plt.xlabel("Month of the Year")
    plt.ylabel("# of transactions")
    plt.subplot(1, 2, 2)
    valid_months.hist(bins=num_months, color="green")
    plt.title("Valid transactions by Month")
    plt.xlabel("Month of the Year")
    plt.ylabel("# of transactions")
    plt.tight_layout()
    figure_name = "figure3.png"
    plt.savefig(str(fig_path / figure_name))
    print(f'LOG: Figure: {figure_name} generated in {fig_path}')
    plt.close()

    # Drop the original datetime columns
    merged_df = merged_df.drop([trans_time_col], axis=1)
    merged_df = merged_df.drop([report_time_col], axis=1)

    # Change the data type of trans_time_hour and trans_time_month to object
    merged_df[f"{trans_time_col}_hour"] = merged_df[f"{trans_time_col}_hour"].astype(
        "object"
    )
    merged_df[f"{trans_time_col}_month"] = merged_df[f"{trans_time_col}_month"].astype(
        "object"
    )

    # Data pre-processing: [part IV]
    # Select proper categorical features and encode them
    # ------------------------------------------------------------------------------------

    # Dictionary to store the count of unique categories in each categorical column
    category_counts = {}

    # Get a list of categorical columns
    categorical_columns = merged_df.columns.difference(
        isnot_categorical_columns
    ).tolist()

    # Check the number of unique categories in each categorical column
    for column in categorical_columns:
        unique_values = merged_df[column].unique()
        num_categories = len(unique_values)
        category_counts[column] = num_categories

    # Drop the column 'eventId' as it has unique values and won't contribute to the analysis
    merged_df = merged_df.drop(["eventId"], axis=1)

    # Update the list of categorical column names after dropping 'eventId'
    categorical_columns.remove("eventId")

    # Calculate the cross table of frequencies for each categorical column
    if verbose:
        for name in categorical_columns:
            cross_table = pd.crosstab(merged_df["fraud"], merged_df[name])
            print("Valid/Fraud transactions by:", name, "\n", cross_table)

    # Data pre-processing: [part V]
    # Data Splitting and dealing with categorical features
    # ------------------------------------------------------------------------------------

    # Before performing any further processing, split the data into train and test sets
    y = merged_df["fraud"]
    X = merged_df.drop(["fraud"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
        random_state=random_state, stratify=y)

    # Rename the target variable in the training set
    y_train = y_train.rename("fraud")

    # Encoding Categorical Columns
    X_train_en = X_train.copy()
    X_test_en = X_test.copy()

    # Initialize the encoders
    encoder_OneHot = ce.OneHotEncoder(cols=["posEntryMode"]) # One-hot encoding 
    encoder_targ = ce.TargetEncoder(return_df=True) # Target encoding
    encoder_ord = ce.OrdinalEncoder(return_df=True) #Ordinary encoding

    # Target Encoding all categorical features in X_train based on the target variable y_train
    for column in categorical_columns:
        if column not in [# Target encoding categorical columns that have a considerable number of categories
            "transactionTime_hour",
            "transactionTime_month",
            "posEntryMode",
        ]:
            X_train_en[column] = encoder_targ.fit_transform(X_train_en[column], y_train)
            X_test_en[column] = encoder_targ.fit_transform(X_test_en[column], y_test)
        elif column in [ #Ordinary encoding the month and hour of transactions features.
            "transactionTime_hour",
            "transactionTime_month",
        ]:
            X_train_en[column] = encoder_ord.fit_transform(X_train_en[column])
            X_test_en[column] = encoder_ord.fit_transform(X_test_en[column])

    # Perform one-hot encoding on the 'posEntryMode' column
    # 'posEntryMode' has only 9 categories which is not high. 
    encoder_OneHot = ce.OneHotEncoder(cols=["posEntryMode"])
    X_train_en = encoder_OneHot.fit_transform(X_train_en)
    X_test_en = encoder_OneHot.fit_transform(X_test_en)

    # Convert all data types to float32 for xgboost training
    X_train_en = X_train_en.astype("float32")
    X_test_en = X_test_en.astype("float32")

    if verbose:
        # Print the data types of X_train_en and display its head
        print(X_train_en.dtypes)
        print(X_train_en.head())

    # Model generation (XGBoost)
    # ------------------------------------------------------------------------------------

    # Generate the Predictive models

    # Convert data into the internal XGBoost data structure
    dmat_train = xgb.DMatrix(X_train_en, y_train)
    dmat_test = xgb.DMatrix(X_test_en, y_test)

    # Set XGBoost parameters
    params = {}
    params["eta"] = 0.02
    params["max_depth"] = 8
    params["subsample"] = 0.5
    params["random_state"] = random_state

    # Train XGBoost model
    xgb_model = xgb.train(
        params, dmat_train, MAX_ROUNDS, maximize=True, verbose_eval=VERBOSE_EVAL
    )

    # Obtain predictions
    dXtest = xgb.DMatrix(X_test_en)
    predictions = xgb_model.predict(dXtest)

    # Convert the predicted probabilities to binary labels based on a threshold
    binary_predictions = (predictions >= threshold).astype(int)

    # Model evaluation
    # ------------------------------------------------------------------------------------

    # Calculate evaluation measures
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    auc_value = roc_auc_score(y_test, predictions)
    aucpr = average_precision_score(y_test, predictions)

    # Print the evaluation measures
    print(" ")
    print("Evaluation Summary:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", auc_value)
    print("AUC PR:", aucpr)
    print(" ")

    # Store the best model to a file in the model directory
    joblib.dump(xgb_model, model_path / "best_model_XGB.pkl")
    print(f'LOG: The best model called best_model_XGB.pkl stored in {model_path} directory')

    # Calculate precision, recall, and F1 score
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, binary_predictions)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    print("")

    # Generating a set of Evaluation plots
    # ------------------------------------------------------------------------------------

    # Plot variable importance
    fig, (ax) = plt.subplots(ncols=1, figsize=(12, 10))
    xgb.plot_importance(
        xgb_model,
        height=0.8,
        title="Feature Importance (XGBoost)",
        ax=ax,
        color="green",
    )
    figure_name = "fig_importance.png"
    plt.tight_layout()
    plt.savefig(str(fig_path / figure_name))
    print(f'LOG: Figure: {figure_name} generated in {fig_path}')
    plt.close()

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)  # Adjust font size for better visualization
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    figure_name = "Confusion_matrix.png"
    plt.savefig(str(fig_path / figure_name))
    print(f'LOG: Figure: {figure_name} generated in {fig_path}')
    plt.close()

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    fig_precicion_recall = plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    figure_name = "fig_precicion_recall.png"
    plt.savefig(str(fig_path / figure_name))
    print(f'LOG: Figure: {figure_name} generated in {fig_path}')
    plt.close()

    # Plot the ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, predictions)
    plt.figure(figsize=(8, 6))
    fig_ROC = plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    figure_name = "fig_ROC.png"
    plt.savefig(str(fig_path / figure_name))
    print(f'LOG: Figure: {figure_name} generated in {fig_path}')
    plt.close()


if __name__ == "__main__":
    # Set the input parameters (Refer to the README file for the description of each parameter)
    # -------------------------------------------------------------------------------------------------------------
    working_path = "/Users/your_directory"
    transactions_fname = "transactions_obf.csv"
    labels_fname = "labels_obf.csv"
    verbose = False
    threshold = 0.5
    test_size = 0.3
    random_state = 2023
    MAX_ROUNDS = 200
    VERBOSE_EVAL = 10

    # Call the process_data function with the argument values
    fraud_XGboost(
        working_path,
        transactions_fname,
        labels_fname,
        verbose=verbose,
        threshold = threshold,
        test_size=test_size,
        random_state=random_state,
        MAX_ROUNDS=MAX_ROUNDS,
        VERBOSE_EVAL=VERBOSE_EVAL,
    )
