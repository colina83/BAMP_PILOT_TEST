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
from Functions.prediction import *
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE, ADASYN


### Function to import data

def import_data(dir):
    """
    This function is used to import data from a csv file.
    
    Parameters:
    dir (str): A string that represents the directory of the csv file and the name of the file. The string must be in the format of "folder/name_of_file.csv".
    
    Returns:
    donors (DataFrame): A pandas dataframe that contains the imported data.
    
    Example:
    donors = import_data("folder/donors.csv")
    """
    donors=pd.read_csv(dir,index_col=False)
    donors.drop(columns="Unnamed: 0",inplace=True)

    print(f"The file you are loading is a:{dir}, with :{donors.shape[0]} observations, and {donors.shape[1]},variable"   )
    return donors 

#Looking for NA's 

def missing_values_table(df):
    """
    This function calculates the number and percentage of missing values in a given pandas dataframe. 

    Parameters:
    df (pandas dataframe): The dataframe to analyze.

    Returns:
    pandas dataframe: A table with two columns: 'Missing Values' and '% of Total Values'.
    The 'Missing Values' column lists the number of missing values for each column in the input dataframe.
    The '% of Total Values' column lists the percentage of missing values for each column in the input dataframe.
    The table is sorted in descending order of the '% of Total Values' column, so that the columns with the most missing values appear first.
    The function also prints a summary statement indicating the total number of columns in the input dataframe, and the number of columns with missing values.

    Example:
    missing_values_table(df)
    > Your selected dataframe has 10 columns.
    > There are 3 columns that have missing values.
    >  
    >    Missing Values   % of Total Values
    >    ColumnA          25.0
    >    ColumnB          20.0
    >    ColumnC          15.0
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns


## Creates a Pie Chart for Binary Variables only

def pie_churn(df,var):
    """
    df -> DataFrame containing the data you want to use
    var -> The variable (single) you want to display on a pie chart    
    """
    print(df[str(var)].value_counts(normalize=True)*100)
    a = plt.pie(df[str(var)].value_counts(),labels=['Not Churned=0','Churned=1'])


def bar_churn(df, var, name = None):
    """
    This function plots a bar chart to visualize the distribution of the target variable (churn) in the input dataframe.

    Parameters:
    df (pandas dataframe): The dataframe containing the data to be used.
    var (string or int): The name or index of the target variable (churn) in the dataframe.
    name (string, optional): A string to add to the title of the plot. The default value is None.

    Returns:
    Matplotlib bar chart: A bar chart showing the distribution of the target variable (churn) in the input dataframe. The x-axis is labeled 
    'Churn Status' and the y-axis is labeled 'Percentage'. The chart displays the percentage of customers who have churned (1) and
     the percentage of customers who have not churned (0).
     If a title string is provided as the 'name' argument, it will be displayed as the title of the plot. 
     If 'name' is not provided, the default title of the plot is 'Churn Distribution'. 
     The chart also displays the percentage value for each bar.

    Example:
    bar_churn(df, 'Churn', 'Churn Distribution of Customers')
    """
    counts = df[str(var)].value_counts(normalize=True)*100
    labels = ['Not Churned (0)', 'Churned (1)']
    plt.bar(labels, counts, color=['g', 'r'], width = 0.5)
    plt.xlabel('Churn Status')
    plt.ylabel('Percentage')
    if name:
        plt.title(name)
    else:
        plt.title('Churn Distribution')
    for i in range(len(counts)):
        plt.text(x = labels[i], y = counts[i]+1, s = str(round(counts[i],2))+'%', size = 10)
    plt.show()

# Plot to explore relationship with donors 
def new_donor_exp(df,var):
    """
    Provides variables you want to analyze
    df = Data 
    var = Variables to Explore in relationship to new donors

    """
    data_exp = pd.DataFrame(df[var])
    df_exp_filtered = data_exp[data_exp["New.Donor"]==1]
    plt.hist(df_exp_filtered['last.appointment'])
    plt.show()

## Code to run cross-validation on different models

def built_models_cv(n, X, y):
    """
    Build several machine learning models using cross validation.

    Parameters:
    n (int): number of validation folds.
    X (array-like): training data.
    y (array-like): target data for the training data.

    Returns:
    pd.DataFrame: A dataframe that contains the models and the number of CV performed. The dataframe has the models as columns and the accuracy score as values.
    """
    random_state = 42 # Define a random seed to ensure reproducibility of results
    
    # Define a dictionary containing the models to be used
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Support Vector Machine': svm.SVC(random_state=random_state),
        'CatBoost': CatBoostClassifier(random_state=random_state, eval_metric="Accuracy", logging_level="Silent"),
        'XGBoost': XGBClassifier(random_state=random_state),
        'Logistic Regression': LogisticRegression(solver='lbfgs', random_state=random_state),
        'Gaussian Naive Bayes': GaussianNB()
    }
    
    # Dictionary to store the cross validation results
    results = {}
    
    # Loop through the models in the classifiers dictionary
    for name, model in classifiers.items():
        # Perform cross validation on each model and store the results in the results dictionary
        scores = cross_val_score(model, X, y, scoring="accuracy", cv=n)
        results[name] = scores
        print(f"Cross validation for {name} completed.")
    
    # Convert the results dictionary to a pandas dataframe
    cv_results = pd.DataFrame.from_dict(results, orient='columns')
    
    return cv_results


# PLots the cross-validation for the models:

def plot_models_results(df):
    """
    Plot the results of a cross-validation model.

    Parameters:
    df (pd.DataFrame): Dataframe containing all models with the correspondent CV.

    Returns:
    None. The function returns a plot.
    """
    colors = ['b', 'g', 'r', 'c', 'y', 'm', 'b']
    
    # Loop through the models in the dataframe and plot their results
    for model, color in zip(df, colors):
        plt.plot(np.arange(1, len(df)+1), df[model], color,
                 label=f'{model} = {round(df[model].mean(), 2)}')
        plt.legend(bbox_to_anchor=(1.1, 1.05))


## Random Search Model

def random_search(model, X, y, params):
    """
    Perform randomized search to find the best hyperparameters for a given model.
    
    Arguments:
        model -- any machine learning model that can be used with sklearn's API
        X -- training features
        y -- training target
        params -- a dictionary of hyperparameters to be tuned
    
    Returns:
        best_params -- a dictionary containing the best hyperparameters found
    """
    # Define the StratifiedKFold cross-validation strategy
    kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    
    # Initialize the RandomizedSearchCV object with the model, hyperparameters, cross-validation strategy, number of iterations, number of parallel jobs, and scoring metric
    random_search = RandomizedSearchCV(model, params, cv=kfold, n_iter=20, n_jobs=-1, scoring='recall')
    
    # Fit the model using the randomized search hyperparameters
    random_search.fit(X, y)
    
    # Get the best score from the randomized search
    score_random = random_search.best_score_
    print("The best score is:", score_random)
    
    # Return the best hyperparameters found by the randomized search
    return random_search.best_params_

def run_final_model(model,params,X_train,X_test,y_train,y_test):
    """
    This function is used to run the final model with the best parameters found from the `random_search` function.
    
    Arguments:
        model -> Any model object, e.g. RandomForestClassifier
        params -> Dictionary of the best parameters found from `random_search`
        X_train -> Features in the training dataset
        X_test -> Features in the test dataset
        y_train -> Response variables in the training dataset
        y_test -> Response variables in the test dataset
        
    Returns:
        y_pred -> Predicted response variables for the test dataset
        cf -> Confusion matrix of the prediction results
        model_fitted -> Fitted model object
    """
    
    # Instantiate the final model using the best parameters
    model_final = model(**params)
    
    # Fit the model on the training data
    model_fitted =  model_final.fit(X_train,y_train)
    
    # Predict the response variables for the test data
    y_pred = model_fitted.predict(X_test)
    
    # Print the accuracy score and classification report
    print('Generalization Score: ',round(accuracy_score(y_test,y_pred),2))
    print(classification_report(y_test,y_pred))
    
    # Calculate the confusion matrix
    cf = confusion_matrix(y_test,y_pred)
    classification_report(y_test,y_pred, target_names= ["Not Churned", "Churned"])
        
    # Return the predicted response variables, the confusion matrix, and the fitted model
    return y_pred,cf,model_fitted

def feature_importance(df,model):  
    """
    This function plots the relative feature importance of a given model
    
    Arguments:
        * df: DataFrame containing the features
        * model: fitted machine learning model with feature importance attributes
        
    Output:
        * Plot of relative feature importance
    """
    
    feature_importance = abs(model.feature_importances_) # get the feature importance values
    feature_importance = 100.0 * (feature_importance / feature_importance.max()) # normalize to 100
    sorted_idx = np.argsort(feature_importance) # sort the features based on importance
    pos = np.arange(sorted_idx.shape[0]) + .5 # set the bar positions

    # create a bar plot of the feature importances
    featfig = plt.figure()
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(df.columns)[sorted_idx], fontsize=8)
    featax.set_xlabel('Relative Feature Importance')

    plt.tight_layout()   
    plt.show()


# CREATES DATASET
def datasets(df, size, x, y):
    """
    df : pandas DataFrame
        DataFrame that contains the data
    size : float
        Size of the test set in the range [0, 1].
    x : list
        List of feature names to be used as inputs.
    y : string
        Name of the target column.
    
    Returns:
        4 elements: x_train, x_test, y_train, y_test.
        x_train : pandas DataFrame
            Training data for the features.
        x_test : pandas DataFrame
            Testing data for the features.
        y_train : pandas Series
            Training data for the target.
        y_test : pandas Series
            Testing data for the target.
    """
    # Split the DataFrame into train and test sets
    train, test = train_test_split(df, test_size=size, random_state=42)
    
    # Store the feature names in 'variables'
    variables = x
    
    # Create the feature datasets (training and testing)
    x_train = train[variables]
    x_test = test[variables]
    
    # Create the target datasets (training and testing)
    y_train = train[y]
    y_test = test[y]
    
    # Return the datasets
    return x_train, x_test, y_train, y_test


## Scaling and Transforming

def pipeline_churn(x_train,x_test):
    """
    x_train -> training data features
    x_test -> test data features
    
    """
    # Selecting the numeric columns from x_train
    numeric_cols = x_train.select_dtypes(exclude="object").columns.tolist()
   
    # Creating a pipeline for scaling the numeric columns
    num_pipeline = Pipeline([        
        ('std_scaler', StandardScaler()), # Standardizing the numeric columns
        ])

    # Final pipeline for transforming both numeric and non-numeric columns
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numeric_cols) # Using the numeric pipeline for numeric columns
    ])
 
    # Transforming the training data 
    X_train_prepared = full_pipeline.fit_transform(x_train)

    # Transforming the test data
    X_test_prepared= full_pipeline.transform(x_test) 
    
    return X_train_prepared,X_test_prepared 



def ETL_pipeline(dir,var_clustering):
    """
    Imports and performs transformations on a dataset for further analysis.

    Parameters:
    dir (str): path of the dataset
    var_clustering (list): list of variables to be used in clustering

    Returns:
    pandas.DataFrame: The cleaned and transformed dataset.

    """
    # Imports Data
    donors = import_data(dir) 

    # Removes Infinite Values
    donors.replace([np.inf, -np.inf], 0, inplace=True) 
    
    ## Creates new variable New_Donation_Adjusted
    # creates a condition where Last.Visit.gap <= 10 and New.Donor == 1
    conditions = [(donors['Last.Visit.gap'] <= 10) & donors['New.Donor'] == 1]
    values = [1]
    # creates a new column 'New_Donation_Adjusted'
    donors['New_Donation_Adjusted'] = np.select(conditions, values)

    # Parameters for the Silhouette Coefficient for KMeans clustering
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
    }   
    
    # subset of variables to be used in clustering
    df_k_means = donors[var_clustering]
    
    # standardize the variables for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_k_means)

    # Store Silhouette Coefficient scores for each number of clusters
    silhouette_coefficients = []

    # Calculate Silhouette Coefficient for number of clusters ranging from 2 to 20
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)

    # Find the number of clusters that provides the highest Silhouette Coefficient score
    max_score = int(max(silhouette_coefficients)*10)
    
    # Fit the KMeans clustering model with the highest Silhouette Coefficient
    kmeans = KMeans(
    init="random",
    n_clusters=max_score,
    n_init=11,
    max_iter=1000,
    random_state=42
    )
    kmeans.fit(scaled_features)
    
    # Creates New Variable 'Clustering'
    donors["Clustering"] = kmeans.labels_
    
    # Returns the transformed dataset
    return donors

def test_data_dir(dir,variables_cluster):
    """
    This function runs the ETL_pipeline function on multiple directories in `dir` and saves the output in a dictionary.

    Parameters:
        dir (list): list of directories containing data
        variables_cluster (list): list of variables for clustering

    Returns:
        validation_dict (dict): dictionary of data frames returned by ETL_pipeline function

    """
    validation_dict = {}
    format = "%Y-%m-%d"

    for i in dir:
        dir_name = str(i).split('_')[1][:-4]
        validation_dict[dir_name] = ETL_pipeline(i,variables_cluster)

    return validation_dict   


def pos_neg_target(df):
    """
    This function calculates the ratio of negative and positive targets in the input data frame `df`.

    Parameters:
        df (pandas.DataFrame): data frame containing target variable

    Returns:
        ratio (float): ratio of negative to positive targets
    
    """
    neg, pos = np.bincount(df['target'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
    return neg/pos


