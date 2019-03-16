import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.tree
import sklearn.ensemble
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(feature_file, label_file):
	"""
	read in feature file and label file for the project
	-----------
	Input:
	1. feature_file: the url or file name for the feature file
	2. label_file; the url or file name for the label file
	Output:
	1. Pandas dataframe for features with id as index
	2. Pandas dataframe for label with id as index
	"""
    df_feature = pd.read_csv(feature_file)
    df_label = pd.read_csv(label_file)
    df_feature.set_index('id', inplace=True)
    df_label.set_index('id', inplace=True)
    return df_feature, df_label

def get_cols_drop():
	"""
	get a list of column names that need to drop
	"""
    cols_drop = ['recorded_by', 'scheme_name', 'funder', 'installer', \
                'wpt_name', 'subvillage', 'ward', 'lga', 'date_recorded']
    return cols_drop

def convert_int_to_str(df):
	"""
	convert integer variables region_code and district_code to categorical variables
	----------
	Input: pandas dataframe
	Ouput: pandas dataframe after converting integer variables region_code and district_code to categorical variables
	"""
    df['region_code'] = df['region_code'].astype('str')
    df['district_code'] = df['district_code'].astype('str')
    return df

def get_cols_dummy():
	"""
	return a list of categorical variables that need to be transformed into dummy variables
	"""
    cols_dummy = ['basin','region','region_code','district_code','public_meeting','scheme_management','permit',\
                  'extraction_type','extraction_type_group','extraction_type_class','management',\
                  'management_group','payment','payment_type','water_quality','quality_group','quantity',\
                  'quantity_group','source','source_type','source_class','waterpoint_type','waterpoint_type_group']
    return cols_dummy

def preprocess_feature(df):
	"""
	drop columns, convert two integer columns to string, create dummy variables for categorical columns
	----------
	Input: pandas dataframe for features
	Ouput: pandas dataframe after the transformation
	"""
    cols_drop = get_cols_drop()
    cat_cols = get_cols_dummy()
    df.drop(cols_drop, axis=1, inplace=True)
    df = convert_int_to_str(df)
    for col in cat_cols:
        cat_df = pd.get_dummies(df[col], prefix = col, prefix_sep='_', dummy_na=True,
                               drop_first=False)
        df = df.drop(col, axis=1)
        df = df.merge(cat_df, how='inner', left_index=True, right_index=True)
        logger.info("dummy col:{}".format(col))
    return df

def build_random_forest(X_train, y_train):
	"""
	Use pipeline to combine select k best features and random forest steps, fit the training data
	Input:
	1. feature variables (pandas dataframe)
	2. label variable (pandas dataframe)
	Output:
	1. a pipeline model that combines the select k best features step and the random forest step
	"""
    select_feature = sklearn.feature_selection.SelectKBest(k=100)
    clf = sklearn.ensemble.RandomForestClassifier()
    steps = [('feature_selection', select_feature),
             ('random_forest', clf)]
    pipeline = sklearn.pipeline.Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_random_forest(X_test, model):
	"""
	Predict the label variables using an existing model for a new dataset
	----------
	Input:
	1. new feature dataset
	2. a classification model
	Output:
	1. The prediction result for the new feature dataset using the input classification model
	"""
    y_predictions = model.predict(X_test)
    return y_predictions

def evaluate_random_forest(y_test, y_pred):
	"""
	evaluation the random forest model by comparing the predicted results with the true results
	----------
	Input: 
	1. predicted label variable
	2. true label variable
	Ouput:
	1. classification report that shows accuracy, precision, recall
	"""
    report = sklearn.metrics.classification_report(y_test, y_pred)
    print(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, help="data file url or csv file path")
    args = parser.parse_args()
    file_list = args.files.split(",")
    logger.info(file_list)
    feature_file = file_list[0]
    label_file = file_list[1]
    df_feature, df_label = read_data(feature_file, label_file)
    df_feature = preprocess_feature(df_feature)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_feature, \
		df_label, test_size=0.2, random_state=42)
    model = build_random_forest(X_train, y_train)
    y_pred = predict_random_forest(X_test, model)
    evaluate_random_forest(y_test, y_pred)

if __name__=="__main__":
	main()