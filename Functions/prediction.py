#Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import FunctionTransformer,OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn import svm
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix,\
 f1_score, recall_score, precision_score, roc_auc_score,roc_curve
import datetime
from scipy.stats import pearsonr
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import warnings
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
#from plotnine import *
from sklearn.decomposition import PCA

## FUNCTIONS TO DO PREDICTIONS

def pipeline_fe(df,var,n):
    """
    A function that receives a dataframe and the number of clusters (n) as inputs, and returns the same dataframe after processing the data.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe
    var (list): List of variables to use for clustering
    n (int): Number of clusters to be generated
    
    Returns:
    df (pandas.DataFrame): The processed dataframe
    
    """
    
    # Create a new donor adjustment
    conditions = [(df['last.appointment'] <= 10) & (df['New.Donor'] == 1)]
    values = [1]
    df['New_Donation_Adjusted'] = np.select(conditions, values)
    
    # Clustering
    df_k_means = df[var]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_k_means)
    
    kmeans = KMeans(
        init="random",
        n_clusters=n,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(scaled_features)
    df["Clustering"] = kmeans.labels_
    
    return df
    

def pipeline_churn_val(x):
    """
    A function that receives a dataframe as input and returns the prepared data for the churn validation.
    
    Parameters:
    x (pandas.DataFrame): The input dataframe
    
    Returns:
    X_train_prepared (numpy.ndarray): The prepared data for the churn validation
    
    """
    numeric_cols = x.select_dtypes(exclude="object").columns.tolist()
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

    # Running final pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numeric_cols)
    ])
    X_train_prepared = full_pipeline.fit_transform(x)
    
    return X_train_prepared

def missclass(df_test, y_val_pred, predicted_prob):
    """
    This function calculates the missclassified donors and returns the dataframes for missclassified churn and missclassified not-churn donors.
    
    Parameters:
    df_test (dataframe) : Test dataset
    y_val_pred (array) : Predicted target values
    predicted_prob (array) : Probability scores for the target class (1 or 0)
    
    Returns:
    missclassified_churn (dataframe) : Dataframe containing the missclassified donors who churned
    missclassified_not_churn (dataframe) : Dataframe containing the missclassified donors who did not churn

    """
    # Create a dataframe for analysis and validation
    d = {
        'Donor.ID': df_test["Donor.ID"],
        'Predicted Churn Definition': y_val_pred,
        'Actual Churn Definition': df_test['target'],
        'Number of Appointments': df_test["Number.Appointments"],
        'First Donation': df_test["First.Date"],
        'Last Donation': df_test["Last.Date"],
        'Gap': df_test["SD.Apointment.Gap"],
        "Risk": df_test["fc.Risk.Def"],
        'Recency': df_test["Recency"],
        "Period": df_test["Donation.Period"],
        'probability': predicted_prob[:, 1],
        'Last Appointment': df_test["last.appointment"]
    }

    may_val = pd.DataFrame(data=d)
    
    # Create a dataframe for missclassified donors
    missclassified = may_val[may_val["Predicted Churn Definition"] != may_val["Actual Churn Definition"]]
    
    # Create dataframes for missclassified churn and not-churn donors
    missclassified_churn = missclassified[missclassified['Actual Churn Definition'] == 1]
    missclassified_not_churn = missclassified[missclassified['Actual Churn Definition'] == 0]
    
    print("Number of Missclassified Donors that Churned:", missclassified_churn.shape)
    print("Number of Missclassified Donors that did not Churn:", missclassified_not_churn.shape)
    
    return missclassified_churn, missclassified_not_churn

def plot_miss(data, varname):
    """
    Plots a histogram of the missing values of a given variable.

    Parameters:
    data (pandas DataFrame): DataFrame containing the data
    varname (str): The name of the column in the data frame to plot
    
    Returns:
    plotly Figure: A plotly histogram of the missing values

    """
    fig = px.histogram(data, x=varname)
    return fig

def run_validations(donors_val, var, model, prob):
    """
    Imports the data and runs a full prediction.

    Parameters:
    donors_val (pandas DataFrame): DataFrame containing the validation data
    var (list of str): The list of variables to be used for validation
    model (keras.engine.sequential.Sequential or scikit-learn model): The model to use for prediction
    prob (float): The threshold to use for binary classification

    Returns:
    x_val (numpy.ndarray): The validation data after processing
    y_val_pred (numpy.ndarray): The binary predictions based on the model
    predicted_prob (numpy.ndarray): The predicted probabilities for each observation
    cm_val (numpy.ndarray): The confusion matrix for the predictions
    report (str): A classification report for the predictions
    donors_val (pandas DataFrame): The validation data

    """
    x_val = pipeline_churn_val(donors_val[var])
    
    if str(type(model)) == "<class 'keras.engine.sequential.Sequential'>":
        predicted_prob = model.predict(x_val)
        y_val_pred = (predicted_prob > prob)    
    else:
        predicted_prob = model.predict_proba(x_val)
        y_val_pred = (predicted_prob[:, 1] >= prob).astype('int')
    
    cm_val = confusion_matrix(donors_val["target"], y_val_pred)
    report = classification_report(donors_val["target"], y_val_pred, target_names=["Not Churned (0)", "Churned (1)"])
    result_dict = {
        "X Validation Observations": x_val.shape[0],
        "X Validation Features": x_val.shape[1]
    }
    print(result_dict)
    return x_val, y_val_pred, predicted_prob, cm_val, report, donors_val


def predictions_dictionary(data_dict, FINAL_VAR, model, prob):
    """
    This function runs the `run_validations` function on each element of the `data_dict` dictionary and returns a dictionary of results.
    
    Parameters:
        data_dict (dict): Dictionary where each key is the name of the data and its value is the data itself.
        FINAL_VAR (list): List of strings representing the final variables to be used in the prediction.
        model (model object): Model object to be used for predictions.
        prob (float): Threshold for the predictions.
        
    Returns:
        results_dict (dict): Dictionary where each key is the name of the data and its value is a dictionary of results from the `run_validations` function.
    """
    results_dict = {} # create an empty dictionary to store the results
    for key, value in data_dict.items(): # loop over the items in the data_dict
        # run the run_validations function and store the results in variables
        x_val, y_val_pred, predicted_prob, cm_val, report, df_test = run_validations(value, FINAL_VAR, model, prob)
        # add the results as a dictionary to the results_dict
        results_dict[key] = {
            'x_val': x_val,
            'y_val': y_val_pred,
            'predicted_prob': predicted_prob,
            'cm_val': cm_val,
            'report': report,
            'df_test': df_test
        }
    return results_dict # return the results_dict
    

def plot_confusion_matrix(dictionary,directory):
    """
    This function plots a heatmap of the confusion matrix for each item in the `directory` list.
    The function accesses the confusion matrix for each item in the `dictionary` by using the item's name
    (extracted from the file name in the `directory` list) as the key.
    
    Parameters:
    dictionary (dict): A dictionary containing the results of the prediction run
    directory (list): A list of file names to be used as keys to access the confusion matrix in the `dictionary`
    
    Returns:
    None (void function)
    """
    for file in directory:
        # Extract the name of the item (presumably a data file) from the file name in the `directory` list
        dir_name = str(file).split('_')[1][:-4]
        
        # Use the item name as a key to access the confusion matrix in the `dictionary`
        cm_val = dictionary[dir_name]["cm_val"]
        
        # Plot the confusion matrix as a heatmap using Seaborn
        sns.heatmap(cm_val, annot=True,fmt='d')