# --- Code Block ---
#package import
import numpy as np
import pandas as pd
import os
import getpass
import teradata
import dask.dataframe as dd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotly.express as px
import json

#get pass
os.getcwd()

#set project directory
path = r"\\file018\PMO_GRPB\RENEWAL\CMS Data\SHDS\AC Replacement Model"

# --- Code Block ---
#create teradata connection

uda = teradata.UdaExec(appName='project_name', version='1.0')
connection = uda.connect(driver='Teradata', 
                 method='odbc', system='TDPROD', username=os.environ.get('user'), 
                 password=getpass.getpass(), authentication="LDAP")

#get all the feature names
ccsr = pd.read_sql("SELECT DISTINCT CCSR_CATEGORY_1 AS CCSR_CATEGORY FROM P_CXS_SBXD.DXCCSR_v2023 ORDER BY CCSR_CATEGORY_1", connection)
ccs = pd.read_sql("SELECT DISTINCT CCS FROM P_CXS_SBXD.CCS_Procedures_v2022 ORDER BY CCS", connection)
drgs = pd.read_sql("SELECT DISTINCT RIGHT(DIAGNOSIS_RELATED_GROUP_INFORMATION,3) DRG FROM P_SDM_ETRAM_STG_VW.V_ETRAM_INSTITUTIONAL_CLAIM ORDER BY DRG WHERE DRG <> 'None'", connection)


startdates = ['2018-01-01']

for startdate in startdates: 

    #policies
    pol_sql = open(path + "\SQL\Train_Policies.sql",'r').read()
    pol_sql = pol_sql.format(StartDate= startdate)
    pol = pd.read_sql(pol_sql,connection)
    pol.columns = pol.columns.str.lower()

    #icd10
    icd_sql = open(path + "\SQL\Train_ICD10CM.sql",'r').read()
    icd_sql = icd_sql.format(StartDate= startdate)
    icd = pd.read_sql(icd_sql,connection)

    #cpt & drg
    cpt_sql = open(path + "\SQL\Train_CPT.sql",'r').read()
    cpt_sql = cpt_sql.format(StartDate= startdate)
    cpt = pd.read_sql(cpt_sql,connection)

    #actual to expected ratio
    ae_sql = open(path + "\SQL\Train_AtoE.sql",'r').read()
    ae_sql = ae_sql.format(StartDate= startdate)
    ae = pd.read_sql(ae_sql,connection)
    ae.columns = ae.columns.str.lower()



    #pivot for icd10
    icd_pivot = icd.pivot_table(index=['PARTY_ID'],
                            columns = ['CCSR_CATEGORY'],  
                            values=['CCSR_3mos', 'CCSR_6mos', 'CCSR_12mos'],
                            aggfunc='sum')

    icd_pivot = icd_pivot.reindex(columns=pd.MultiIndex.from_product([icd_pivot.columns.levels[0], ccsr['CCSR_CATEGORY']], names=icd_pivot.columns.names), fill_value=0)
    icd_pivot.columns = ['{}_{}'.format(col[0], col[1]) for col in icd_pivot.columns]
    icd_pivot.reset_index(inplace=True)
    icd_pivot.columns = icd_pivot.columns.str.lower()


    #pivot for drg
    drg = cpt[~cpt['DRG'].isna()][['PARTY_ID','DRG','DRG_LoS']]
    drg_pivot = drg.pivot_table(index=['PARTY_ID'],
                            columns = ['DRG'],  
                            values=['DRG_LoS'],
                            aggfunc='sum')

    drg_pivot = drg_pivot.reindex(columns=pd.MultiIndex.from_product([drg_pivot.columns.levels[0], drgs['DRG']], names=drg_pivot.columns.names), fill_value=0)
    drg_pivot.columns = ['{}_{}'.format(col[0], col[1]) for col in drg_pivot.columns]
    drg_pivot.reset_index(inplace=True)
    drg_pivot.columns = drg_pivot.columns.str.lower()

    #pivot for cpt
    cpt = cpt[~cpt['CCS'].isna()][['PARTY_ID','CCS','CCS_3mos','CCS_6mos','CCS_12mos']]

    cpt_pivot = cpt.pivot_table(index=['PARTY_ID'],
                            columns = ['CCS'],  
                            values=['CCS_3mos', 'CCS_6mos', 'CCS_12mos'],
                            aggfunc='sum')
    cpt_pivot = cpt_pivot.reindex(columns=pd.MultiIndex.from_product([cpt_pivot.columns.levels[0], ccs['CCS']], names=cpt_pivot.columns.names), fill_value=0)
    cpt_pivot.columns = ['{}_{}'.format(col[0], col[1]) for col in cpt_pivot.columns]
    cpt_pivot.reset_index(inplace=True)
    cpt_pivot.columns = cpt_pivot.columns.str.lower()


    #merge all the dfs
    pol = pd.merge(pol, icd_pivot, how = 'left', on = 'party_id')
    pol = pd.merge(pol, drg_pivot, how = 'left', on = 'party_id')
    pol = pd.merge(pol, cpt_pivot, how = 'left', on = 'party_id')
    pol = pd.merge(pol, ae, how = 'inner', on = 'party_id')

    pol['startdate'] = startdate
    pol.to_feather(path + '\Data\\' + 'train ' + startdate + '.feather')

connection.close()

# --- Code Block ---
# Get a list of all feather files in the directory
files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('.feather')]

ae_stats = []

#read the feather file
for file in files:
    filepath = os.path.join(path + '\Data\\', file)
    df = pd.read_feather(filepath)


    #downcast float and int column
    for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
    #remove trailing spaces
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        
    df = df[df['plan'].isin(['F', 'G', 'N'])]
    df.drop(columns=['policy_agreement_id'],inplace=True)

    #Apply the mapping to the 'zip' column in the new DataFrame
    file_path = 'zip_to_category.json'

    # Import the dictionary from the zip JSON file
    with open(file_path, 'r') as f:
        zip_to_category = json.load(f)

    #map zip to zip_category
    df['zip_category'] = df['zip'].map(zip_to_category)

    #ae descriptive statistics
    ae_stat=df[['a_to_e_value']].describe().transpose()
    ae_stat['file'] = file
    ae_stats.append(ae_stat)

    for column in ['state', 'plan', 'gender', 'tobacco_usage_ind',
       'marketing_channel', 'uw_catg_des', 'hhd_ind','zip_category']:
        # Group by the current column to calculate count and average 'a_to_e_value'
        grouped_data = df.groupby(column)['a_to_e_value']
        count = grouped_data.count()
        average = grouped_data.mean()

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(24, 6))

        # Plot count
        axs[0].bar(count.index, count.values)
        axs[0].set_title('Count by ' + column + file)
        axs[0].set_xlabel(column)
        axs[0].set_ylabel('Count')

        # Plot average
        axs[1].bar(average.index, average.values)
        axs[1].set_title('Average a_to_e_value by ' + column + file)
        axs[1].set_xlabel(column)
        axs[1].set_ylabel('Average a_to_e_value')

        plt.tight_layout()
        plt.show()

# --- Code Block ---
#create teradata connection

uda = teradata.UdaExec(appName='project_name', version='1.0')
connection = uda.connect(driver='Teradata', 
                 method='odbc', system='TDPROD', username=os.environ.get('user'), 
                 password=getpass.getpass(), authentication="LDAP")

#get all the feature names
ccsr_label = pd.read_sql("SELECT DISTINCT CCSR_CATEGORY_1, CCSR_CATEGORY_1_DESCRIPTION AS CCSR_CATEGORY FROM P_CXS_SBXD.DXCCSR_v2023 ORDER BY CCSR_CATEGORY_1", connection)
ccs_label = pd.read_sql("SELECT DISTINCT * FROM P_CXS_SBXD.CCS_Procedures_v2022 ORDER BY CCS", connection, columns=[['CCS','CCS Label']])
ccs_label = ccs_label.drop(columns=['Code']).drop_duplicates()

#Create mappings from the 'ccs_label' and 'ccsr_label' dataframes
ccs_mapping = dict(zip(ccs_label['CCS'].astype(str), ccs_label['CCS Label']))
ccsr_mapping = dict(zip(ccsr_label['CCSR_CATEGORY_1'], ccsr_label['CCSR_CATEGORY']))

# --- Code Block ---
df = pd.read_feather(path + '\Data\\train 2022-01-01.feather')
df.fillna(0, inplace=True)

# Read the contents of the file back into a list
with open('drop_features.txt', 'r') as file:
    drop_features = [line.strip() for line in file]
df = df.drop(columns=drop_features, errors = 'ignore')

#remove trailing spaces
for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

#Apply the mapping to the 'zip' column in the new DataFrame
file_path = 'zip_to_category.json'

# Import the dictionary from the zip JSON file
with open(file_path, 'r') as f:
    zip_to_category = json.load(f)

#map zip to zip_category
df['zip_category'] = df['zip'].map(zip_to_category)

df.drop(columns=['party_id','policy_agreement_id','zip','startdate'],inplace=True)

# --- Code Block ---
'''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Set up preprocessing for numeric columns: scaling
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Set up preprocessing for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(sparse=True, handle_unknown='ignore')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(df.sample(frac=0.1, weights='high_ae', random_state=101))

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_processed)'''

# --- Code Block ---
#package import
import numpy as np
import pandas as pd
import os
import getpass
import teradata
import dask.dataframe as dd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotly.express as px
import json
from scipy.sparse import csr_matrix
import pickle

#get pass
os.getcwd()

#set project directory
path = r"\\file018\PMO_GRPB\RENEWAL\CMS Data\SHDS\AC Replacement Model"

# --- Code Block ---
'''import os
import pandas as pd
from scipy.sparse import csr_matrix
import pickle

def load_features_to_drop():
    """ Load features with 100% missing values or zero correlations. """
    with open('drop_features.txt', 'r') as file:
        nan_features = [line.strip() for line in file]
    
    corr = pd.read_csv('all_correlations.csv')
    corr.rename(columns={'Unnamed: 0': 'feature'}, inplace=True)
    corr_features = corr[corr.isna().any(axis=1)]['feature'].tolist()

    return nan_features + corr_features

def get_us_states():
    """ List of US state abbreviations. """
    return [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]

def preprocess_data(df, drop_features, us_states):
    """ Clean and preprocess dataframe. """
    df['state'] = df['state'].apply(lambda x: x if x in us_states else 'unknown')
    df.drop(columns=['party_id', 'policy_agreement_id', 'zip', 'startdate'], inplace=True)

    df.update(df.select_dtypes(include=['float', 'int']).apply(pd.to_numeric, downcast='integer'))
    df.update(df.select_dtypes(include=['object']).apply(lambda col: col.str.strip().astype(str)))

    df = df[df['plan'].isin(['F', 'G', 'N'])]
    df.drop(columns=drop_features, errors='ignore', inplace=True)
    df.fillna(0, inplace=True)

    y = df.pop('a_to_e_value')
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.concat([df, pd.get_dummies(df[categorical_cols], drop_first=True)], axis=1)

    return csr_matrix(df, dtype='int8'), y

def save_data(df, y, path, index):
    """ Save processed data to files. """
    X_file_name = f"X_sparse{index}.pkl"
    y_file_name = f"y_sparse{index}.pkl"

    with open(os.path.join(path, X_file_name), 'wb') as f:
        pickle.dump(df, f)
    with open(os.path.join(path, y_file_name), 'wb') as f:
        pickle.dump(y, f)

def main():
    
    files = [f for f in os.listdir(os.path.join(path, 'Data')) if f.endswith('01.feather')]
    drop_features = load_features_to_drop()
    us_states = get_us_states()

    for i, file in enumerate(files, start=1):
        df = pd.read_feather(os.path.join(path, 'Data', file))
        df, y = preprocess_data(df, drop_features, us_states)
        save_data(df, y, os.path.join(path, 'Data'), i)

if __name__ == "__main__":
    main()'''

# --- Code Block ---
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import optuna
import pickle
from scipy.sparse import csr_matrix

# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

# Define the sliding window size for training
window_size = 2 #fold 5 is the hold-out set. saving fold 5 for final testing


def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 0.001, 0.3)
    
    # Initialize dictionaries to store metrics
    objective_metric = {'rmse': []}
    
    # Sliding window validation
    for i in range(len(file_names)- window_size):
        # Load dataframes for the current training window
        train_df = pd.read_feather(directory_path + file_names[i])
        test_df = pd.read_feather(directory_path + file_names[i+1])
        
        X_train = train_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value']) #removed gender, and zip per SME
        y_train = train_df['a_to_e_value']
        X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
        y_test = test_df['a_to_e_value']

        
        #Define numeric and categorial features
        numeric_features = X_train.select_dtypes(include=['number']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        #Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
        categorical_transformer = OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

        model = Lasso(alpha=alpha)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predictions for both training and testing to calculate metrics
        predictions_train = pipeline.predict(X_train)
        predictions_test = pipeline.predict(X_test)

        
        # Store metrics for each sliding window in the trial's user attributes
        trial.set_user_attr(f"sw{i+1}_train_mse", mean_squared_error(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_train_rmse", sqrt(mean_squared_error(y_train, predictions_train)))
        trial.set_user_attr(f"sw{i+1}_train_r2", r2_score(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_test_mse", mean_squared_error(y_test, predictions_test))
        trial.set_user_attr(f"sw{i+1}_test_rmse", sqrt(mean_squared_error(y_test, predictions_test)))
        trial.set_user_attr(f"sw{i+1}_test_r2", r2_score(y_test, predictions_test))


        #objective metric
        mse_test = mean_squared_error(y_test, predictions_test)
        rmse_test = sqrt(mse_test)
        objective_metric['rmse'].append(rmse_test)
        


    # Compute and return the average RMSE across all testing splits
    average_rmse = np.mean(objective_metric['rmse'])
    return average_rmse

# Create Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# --- Code Block ---
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

plot_parallel_coordinate(study)

# --- Code Block ---
import plotly.express as px

fig = px.line(results_df, x="number", y=["average_train_rmse","average_test_rmse"], 
              title='Train RMSE vs Test RMSE', 
              labels={"value": "RMSE", "variable": "RMSE Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")
fig.show()

# --- Code Block ---
#retrain lasso with best hyper-parameters
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import optuna
import pickle
from scipy.sparse import csr_matrix

# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

# Define the sliding window size for training
window_size = 1


def objective(trial):
    #alpha = trial.suggest_loguniform('alpha', 0.01, 1)
    # Load Lasso parameters
    ls = pd.read_csv('Lasso_all_feas_06.20.csv')
    best_alpha = ls.sort_values(by='value')['params_alpha'].values[0]
    
    # Initialize dictionaries to store metrics
    objective_metric = {'rmse': []}
    
    # Sliding window validation
    for i in range(len(file_names)- window_size):
        # Load dataframes for the current training window
        train_df = pd.read_feather(directory_path + file_names[i]) #change fold 4
        test_df = pd.read_feather(directory_path + file_names[i+1]) #change to fold 5
        
        X_train = train_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
        y_train = train_df['a_to_e_value']
        X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
        y_test = test_df['a_to_e_value']

        
        #Define numeric and categorial features
        numeric_features = X_train.select_dtypes(include=['number']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        #Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
        categorical_transformer = OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

        model = Lasso(alpha=best_alpha)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predictions for both training and testing to calculate metrics
        predictions_train = pipeline.predict(X_train)
        predictions_test = pipeline.predict(X_test)

        
        # Store metrics for each sliding window in the trial's user attributes
        trial.set_user_attr(f"sw{i+1}_train_mse", mean_squared_error(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_train_rmse", sqrt(mean_squared_error(y_train, predictions_train)))
        trial.set_user_attr(f"sw{i+1}_train_r2", r2_score(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_test_mse", mean_squared_error(y_test, predictions_test))
        trial.set_user_attr(f"sw{i+1}_test_rmse", sqrt(mean_squared_error(y_test, predictions_test)))
        trial.set_user_attr(f"sw{i+1}_test_r2", r2_score(y_test, predictions_test))


        #objective metric
        mse_test = mean_squared_error(y_test, predictions_test)
        rmse_test = sqrt(mse_test)
        objective_metric['rmse'].append(rmse_test)
        
        # Save the trained pipeline to a file
        with open('trained_pipeline_lasso_final.pkl', 'wb') as file:
            pickle.dump(pipeline, file)

    # Compute and return the average RMSE across all testing splits
    average_rmse = np.mean(objective_metric['rmse'])
    return average_rmse

# Create Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)

# --- Code Block ---
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
print(r2_score(y_test, y_ls_pred))
print(sqrt(mean_squared_error(y_test, y_ls_pred)))
print(np.mean(y_ls_pred - y_test))

# --- Code Block ---
import plotly.express as px

# Creating the DataFrame for plotting
data = pd.DataFrame({
    'Measurement': ['y_test', 'y_ls_pred'],
    'Average': [y_test.mean(), y_ls_pred.mean()]
})

# Plot using Plotly Express
fig = px.bar(data, x='Measurement', y='Average', text='Average',
                 title='Lasso Model - Comparison of Averages: y_test vs. y_ls_pred',
                 labels={'Measurement': 'Variable', 'Average': 'Average Value'})

# Show the plot
fig.show()

# --- Code Block ---
# Generate thresholds with a step of 0.1
thresholds = [round(x * 0.1, 1) for x in range(0, 21)]  # This covers from 0.0 to 2.0
def ae_group(average_a_to_e):
    # Find the right interval for the average_a_to_e
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= average_a_to_e < thresholds[i+1]:
            return f'{thresholds[i]} - {thresholds[i+1]}'
    # Handle the case where the value is greater than or equal to the last threshold
    return f'{thresholds[-1]}+'

# --- Code Block ---
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import optuna
import pickle
from scipy.sparse import csr_matrix

# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

# Define the sliding window size for training
window_size = 1


def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 0.01, 1)
    
    # Initialize dictionaries to store metrics
    objective_metric = {'rmse': []}
    
    # Sliding window validation
    for i in range(len(file_names)- window_size):
        # Load dataframes for the current training window
        train_df = pd.read_feather(directory_path + file_names[i])
        test_df = pd.read_feather(directory_path + file_names[i+1])
        
        X_train = train_df.filter(regex='^(ccs|ccsr|drg)')
        y_train = train_df['a_to_e_value']
        X_test = test_df.filter(regex='^(ccs|ccsr|drg)')
        y_test = test_df['a_to_e_value']

        
        #Define numeric and categorial features
        numeric_features = X_train.select_dtypes(include=['number']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        #Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
        categorical_transformer = OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

        model = Lasso(alpha=alpha)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predictions for both training and testing to calculate metrics
        predictions_train = pipeline.predict(X_train)
        predictions_test = pipeline.predict(X_test)

        
        # Store metrics for each sliding window in the trial's user attributes
        trial.set_user_attr(f"sw{i+1}_train_mse", mean_squared_error(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_train_rmse", sqrt(mean_squared_error(y_train, predictions_train)))
        trial.set_user_attr(f"sw{i+1}_train_r2", r2_score(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_test_mse", mean_squared_error(y_test, predictions_test))
        trial.set_user_attr(f"sw{i+1}_test_rmse", sqrt(mean_squared_error(y_test, predictions_test)))
        trial.set_user_attr(f"sw{i+1}_test_r2", r2_score(y_test, predictions_test))


        #objective metric
        mse_test = mean_squared_error(y_test, predictions_test)
        rmse_test = sqrt(mse_test)
        objective_metric['rmse'].append(rmse_test)
        


    # Compute and return the average RMSE across all testing splits
    average_rmse = np.mean(objective_metric['rmse'])
    return average_rmse

# Create Optuna study and optimize
study2 = optuna.create_study(direction='minimize')
study2.optimize(objective, n_trials=25)

# --- Code Block ---
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import optuna
import pickle
from scipy.sparse import csr_matrix

# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

# Define the sliding window size for training
window_size = 1


def objective(trial):
    # LightGBM parameters
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.3),
        'n_estimators': 10000,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    }
    
    # Initialize dictionaries to store metrics
    objective_metric = {'rmse': []}
    
    # Sliding window validation
    for i in range(len(file_names)- window_size):
        # Load dataframes for the current training window
        train_df = pd.read_feather(directory_path + file_names[i])
        test_df = pd.read_feather(directory_path + file_names[i+1])
        
        X_train = train_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
        y_train = train_df['a_to_e_value']
        X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
        y_test = test_df['a_to_e_value']

        
        #Define numeric and categorial features
        numeric_features = X_train.select_dtypes(include=['number']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        #Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
        categorical_transformer = OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

        # Initialize LightGBM model
        model = LGBMRegressor(**param)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predictions for both training and testing to calculate metrics
        predictions_train = pipeline.predict(X_train)
        predictions_test = pipeline.predict(X_test)

        
        # Store metrics for each sliding window in the trial's user attributes
        trial.set_user_attr(f"sw{i+1}_train_mse", mean_squared_error(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_train_rmse", sqrt(mean_squared_error(y_train, predictions_train)))
        trial.set_user_attr(f"sw{i+1}_train_r2", r2_score(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_test_mse", mean_squared_error(y_test, predictions_test))
        trial.set_user_attr(f"sw{i+1}_test_rmse", sqrt(mean_squared_error(y_test, predictions_test)))
        trial.set_user_attr(f"sw{i+1}_test_r2", r2_score(y_test, predictions_test))


        #objective metric
        mse_test = mean_squared_error(y_test, predictions_test)
        rmse_test = sqrt(mse_test)
        objective_metric['rmse'].append(rmse_test)
        


    # Compute and return the average RMSE across all testing splits
    average_rmse = np.mean(objective_metric['rmse'])
    return average_rmse

# Create Optuna study and optimize
study3 = optuna.create_study(direction='minimize')
study3.optimize(objective, n_trials=30)

# --- Code Block ---
#retrain with best hyper-parameters

from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import optuna
import pickle
from scipy.sparse import csr_matrix

# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

# Define the sliding window size for training
window_size = 1


def objective(trial):

    
    # Initialize dictionaries to store metrics
    objective_metric = {'rmse': []}
    
    # Sliding window validation
    for i in range(len(file_names)- window_size):
        # Load dataframes for the current training window
        train_df = pd.read_feather(directory_path + file_names[i])
        test_df = pd.read_feather(directory_path + file_names[i+1])
        
        X_train = train_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
        y_train = train_df['a_to_e_value']
        X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
        y_test = test_df['a_to_e_value']

        
        #Define numeric and categorial features
        numeric_features = X_train.select_dtypes(include=['number']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        #Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
        categorical_transformer = OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

        # Initialize LightGBM model
        model = LGBMRegressor(**best_params)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predictions for both training and testing to calculate metrics
        predictions_train = pipeline.predict(X_train)
        predictions_test = pipeline.predict(X_test)

        
        # Store metrics for each sliding window in the trial's user attributes
        trial.set_user_attr(f"sw{i+1}_train_mse", mean_squared_error(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_train_rmse", sqrt(mean_squared_error(y_train, predictions_train)))
        trial.set_user_attr(f"sw{i+1}_train_r2", r2_score(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_test_mse", mean_squared_error(y_test, predictions_test))
        trial.set_user_attr(f"sw{i+1}_test_rmse", sqrt(mean_squared_error(y_test, predictions_test)))
        trial.set_user_attr(f"sw{i+1}_test_r2", r2_score(y_test, predictions_test))


        #objective metric
        mse_test = mean_squared_error(y_test, predictions_test)
        rmse_test = sqrt(mse_test)
        objective_metric['rmse'].append(rmse_test)
        
        # Save the trained pipeline to a file
        with open('trained_pipeline_lgbm.pkl', 'wb') as file:
            pickle.dump(pipeline, file)


    # Compute and return the average RMSE across all testing splits
    average_rmse = np.mean(objective_metric['rmse'])
    return average_rmse

# Create Optuna study and optimize
study3 = optuna.create_study(direction='minimize')
study3.optimize(objective, n_trials=1)

# --- Code Block ---
import plotly.express as px

# Creating the DataFrame for plotting
data = pd.DataFrame({
    'Measurement': ['y_test', 'y_lasso', 'y_lgbm'],
    'Average': [y_test.mean(), y_ls_pred.mean(), y_pred.mean()]
})

# Plot using Plotly Express
fig = px.bar(data, x='Measurement', y='Average', text='Average',
                 title='Model - Comparison of Averages: y_test vs. y_lasso vs. y_lgbm',
                 labels={'Measurement': 'Variable', 'Average': 'Average Value'})

# Show the plot
fig.show()

# --- Code Block ---
averages = decile.groupby('ae_group').mean().reset_index()
group_counts = decile.groupby('ae_group').size().reset_index(name='count')


# Creating the plot with dual-axis
import plotly.graph_objects as go
fig = go.Figure()

# Adding line charts for averages
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_test'], name='Average y_test',
                         mode='lines+markers', yaxis='y2', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_lasso'], name='Average y_lasso',
                         mode='lines+markers', yaxis='y2', line=dict(color='red')))
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_lgbm'], name='Average y_lgbm',
                         mode='lines+markers', yaxis='y2', line=dict(color='green')))

# Adding bar chart for counts
fig.add_trace(go.Bar(x=group_counts['ae_group'], y=group_counts['count'], name='Count',
                     marker=dict(color='grey'), yaxis='y1'))

# Layout with secondary y-axis for line charts
fig.update_layout(
    title='Average Values and Counts per AE Group',
    xaxis_title='AE Group',
    yaxis=dict(title='Count', side='right'),
    yaxis2=dict(title='Average Value', overlaying='y', side='left'),
    legend_title='Metrics',
    template='plotly_white'
)

# Show the figure
fig.show()

# --- Code Block ---
import seaborn as sns
import matplotlib.pyplot as plt

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(decile.corr(), annot=True, cmap='crest', linewidths=0.5, linecolor='black', fmt=".2f", square=True)
plt.title('Heatmap of Correlation Matrix')

# Show the plot
plt.show()

# --- Code Block ---
# Creating the Plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lasso, mode='lines+markers',
                         name='Cumulative Acceptance Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lgbm, mode='lines+markers',
                         name='Cumulative Acceptance LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_true, mode='lines+markers',
                         name='Cumulative Acceptance True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))
fig.update_layout(
    title='Cumulative Acceptance for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Cumulative Acceptance',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
# Creating the Plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lasso_percent, mode='lines+markers',
                         name='Cumulative Acceptance Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lgbm_percent, mode='lines+markers',
                         name='Cumulative Acceptance LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_true_percent, mode='lines+markers',
                         name='Cumulative Acceptance True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))
fig.update_layout(
    title='Cumulative Acceptance (%) for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Cumulative Acceptance(%)',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
import plotly.graph_objects as go
thresholds = np.linspace(0, 2, 300)

# Calculating cumulative sums for each threshold for each model
cumulative_sums_lasso = [decile[decile['y_test'] < t]['y_lasso'].mean() for t in thresholds]
cumulative_sums_lgbm = [decile[decile['y_test'] < t]['y_lgbm'].mean() for t in thresholds]
#cumulative_sums_true = [decile[decile['y_test'] < t]['y_test'].mean() for t in thresholds]

# Creating the Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
'''fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_true, mode='lines+markers',
                         name='Cumulative Mean True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))'''
fig.update_layout(
    title='Mean a_to_e Values for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
X = uw[['CLAIMS_LOSS_RATIO_SCORE_NUM']]
y = uw['A_TO_E_VALUE']
reg = LGBMRegressor().fit(X, y)
y_uw = reg.predict(X)

# --- Code Block ---
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import optuna
import pickle
from scipy.sparse import csr_matrix

# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

# Define the sliding window size for training
window_size = 1


def objective(trial):
    # LightGBM parameters
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.3),
        'n_estimators': 10000,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    }
    
    # Initialize dictionaries to store metrics
    objective_metric = {'rmse': []}
    
    # Sliding window validation
    for i in range(len(file_names)- window_size):
        # Load dataframes for the current training window
        train_df = pd.read_feather(directory_path + file_names[i])
        test_df = pd.read_feather(directory_path + file_names[i+1])
        
        X_train = train_df.filter(regex='^(ccs|ccsr|drg)')
        y_train = train_df['a_to_e_value']
        X_test = test_df.filter(regex='^(ccs|ccsr|drg)')
        y_test = test_df['a_to_e_value']

        
        #Define numeric and categorial features
        numeric_features = X_train.select_dtypes(include=['number']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        #Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
        categorical_transformer = OneHotEncoder(drop='first', sparse=True, handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

        # Initialize LightGBM model
        model = LGBMRegressor(**param)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predictions for both training and testing to calculate metrics
        predictions_train = pipeline.predict(X_train)
        predictions_test = pipeline.predict(X_test)

        
        # Store metrics for each sliding window in the trial's user attributes
        trial.set_user_attr(f"sw{i+1}_train_mse", mean_squared_error(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_train_rmse", sqrt(mean_squared_error(y_train, predictions_train)))
        trial.set_user_attr(f"sw{i+1}_train_r2", r2_score(y_train, predictions_train))
        trial.set_user_attr(f"sw{i+1}_test_mse", mean_squared_error(y_test, predictions_test))
        trial.set_user_attr(f"sw{i+1}_test_rmse", sqrt(mean_squared_error(y_test, predictions_test)))
        trial.set_user_attr(f"sw{i+1}_test_r2", r2_score(y_test, predictions_test))


        #objective metric
        mse_test = mean_squared_error(y_test, predictions_test)
        rmse_test = sqrt(mse_test)
        objective_metric['rmse'].append(rmse_test)
        


    # Compute and return the average RMSE across all testing splits
    average_rmse = np.mean(objective_metric['rmse'])
    return average_rmse

# Create Optuna study and optimize
study4 = optuna.create_study(direction='minimize')
study4.optimize(objective, n_trials=30)

# --- Code Block ---
#package import
import numpy as np
import pandas as pd
import os
import getpass
import teradata
import dask.dataframe as dd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotly.express as px
import json
from scipy.sparse import csr_matrix
import pickle

#get pass
os.getcwd()

#set project directory
path = r"\\file018\PMO_GRPB\RENEWAL\CMS Data\SHDS\AC Replacement Model"

# --- Code Block ---
import lightgbm as lgb
print(lgb.__version__)
!pip show teradata

# --- Code Block ---
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pickle


# Load Lasso parameters
ls = pd.read_csv('Lasso_all_feas_06.26.csv')
best_alpha = ls.sort_values(by='value')['params_alpha'].values[0]    

#training data
y_train = X_train['a_to_e_value']
X_train = X_train.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])


        
#Define numeric and categorial features
numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

#Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
categorical_transformer = OneHotEncoder(drop='first', 
                                        #sparse=True, 
                                        handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

model = Lasso(alpha=best_alpha)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model
pipeline.fit(X_train, y_train)

# --- Code Block ---
#retrain 

from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
from scipy.sparse import csr_matrix


        
#Define numeric and categorial features
numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

#Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
categorical_transformer = OneHotEncoder(drop='first', 
                                        #sparse=True, 
                                        handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

# Initialize LightGBM model
model = LGBMRegressor(**best_params)

lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model
lgbm_pipeline.fit(X_train, y_train)

# --- Code Block ---
import seaborn as sns
sns.histplot(y_test, kde=True)

# --- Code Block ---
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

print(f"lasso R2: ", r2_score(y_test, y_ls_pred))
print(f"lasso RMSE: ", sqrt(mean_squared_error(y_test, y_ls_pred)))
print(f"lasso ME: ", np.mean(y_ls_pred - y_test))
print("\n")
print(f"lgbm R2: ", r2_score(y_test, y_pred))
print(f"lgbm RMSE: ", sqrt(mean_squared_error(y_test, y_pred)))
print(f"lgbmo ME: ", np.mean(y_pred - y_test))

# --- Code Block ---
import plotly.express as px

# Creating the DataFrame for plotting
data = pd.DataFrame({
    'Measurement': ['y_test', 'y_lasso', 'y_lgbm'],
    'Average': [y_test.mean(), y_ls_pred.mean(), y_pred.mean()]
})

# Plot using Plotly Express
fig = px.bar(data, x='Measurement', y='Average', text='Average',
                 title='Model - Comparison of Averages: y_test vs. y_lasso vs. y_lgbm',
                 labels={'Measurement': 'Variable', 'Average': 'Average Value'})

# Show the plot
fig.show()

# --- Code Block ---
# Generate thresholds with a step of 0.1
thresholds = [round(x * 0.1, 1) for x in range(0, 21)]  # This covers from 0.0 to 2.0
def ae_group(average_a_to_e):
    # Find the right interval for the average_a_to_e
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= average_a_to_e < thresholds[i+1]:
            return f'{thresholds[i]} - {thresholds[i+1]}'
    # Handle the case where the value is greater than or equal to the last threshold
    return f'{thresholds[-1]}+'

# --- Code Block ---
averages = decile.groupby('ae_group').mean().reset_index()
group_counts = decile.groupby('ae_group').size().reset_index(name='count')


# Creating the plot with dual-axis
import plotly.graph_objects as go
fig = go.Figure()

# Adding line charts for averages
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_test'], name='Average y_test',
                         mode='lines+markers', yaxis='y2', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_lasso'], name='Average y_lasso',
                         mode='lines+markers', yaxis='y2', line=dict(color='red')))
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_lgbm'], name='Average y_lgbm',
                         mode='lines+markers', yaxis='y2', line=dict(color='green')))

# Adding bar chart for counts
fig.add_trace(go.Bar(x=group_counts['ae_group'], y=group_counts['count'], name='Count',
                     marker=dict(color='grey'), yaxis='y1'))

# Layout with secondary y-axis for line charts
fig.update_layout(
    title='Average Values and Counts per AE Group',
    xaxis_title='AE Group',
    yaxis=dict(title='Count', side='right'),
    yaxis2=dict(title='Average Value', overlaying='y', side='left'),
    legend_title='Metrics',
    template='plotly_white'
)

# Show the figure
fig.show()

# --- Code Block ---
import seaborn as sns
import matplotlib.pyplot as plt

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(decile[['y_test','y_lasso','y_lgbm']].corr(), annot=True, cmap='crest', linewidths=0.5, linecolor='black', fmt=".2f", square=True)
plt.title('Heatmap of Correlation Matrix')

# Show the plot
plt.show()

# --- Code Block ---
# Creating the Plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lasso, mode='lines+markers',
                         name='Cumulative Acceptance Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lgbm, mode='lines+markers',
                         name='Cumulative Acceptance LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_true, mode='lines+markers',
                         name='Cumulative Acceptance True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))
fig.update_layout(
    title='Cumulative Acceptance for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Cumulative Acceptance',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
# Creating the Plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lasso_percent, mode='lines+markers',
                         name='Cumulative Acceptance Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lgbm_percent, mode='lines+markers',
                         name='Cumulative Acceptance LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_true_percent, mode='lines+markers',
                         name='Cumulative Acceptance True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))
fig.update_layout(
    title='Cumulative Acceptance (%) for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Cumulative Acceptance(%)',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
import plotly.graph_objects as go
thresholds = np.linspace(0, 2, 300)

# Calculating cumulative sums for each threshold for each model
cumulative_sums_lasso = [decile[decile['y_lasso'] < t]['y_test'].mean() for t in thresholds]
cumulative_sums_lgbm = [decile[decile['y_lgbm'] < t]['y_test'].mean() for t in thresholds]
#cumulative_sums_true = [decile[decile['y_test'] < t]['y_test'].mean() for t in thresholds]

# Creating the Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
'''fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_true, mode='lines+markers',
                         name='Cumulative Mean True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))'''
fig.update_layout(
    title='Mean a_to_e Values for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
X = uw[['CLAIMS_LOSS_RATIO_SCORE_NUM']]
y = uw['A_TO_E_VALUE']
reg = LGBMRegressor().fit(X, y)
y_uw = reg.predict(X)

# --- Code Block ---
#package import
import numpy as np
import pandas as pd
import os
import dask.dataframe as dd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotly.express as px
import json


#set project directory
path = r"\\file018\PMO_GRPB\RENEWAL\CMS Data\SHDS\AC Replacement Model"

# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

# --- Code Block ---
import pickle
with open('lasso_model.pkl', 'rb') as file:
    ls_model = pickle.load(file)

y_ls_pred = ls_model.predict(X_test)
test_df['y_pred'] = y_ls_pred

# --- Code Block ---
import getpass
import teradata

sql = "SEL PI.PARTY_ID, PI.FIRST_NAME, PI.LAST_NAME, VPD.POST_CODE, MARITAL_STATUS_NME FROM P_IPA_BI_VW.V_PRIMARY_INSURED_DIM PI INNER JOIN P_IPA_BI_VW.V_PERSON_DIMENSION VPD ON PI.PARTY_ID = VPD.PARTY_ID WHERE PI.BIRTH_DATE IS NOT NULL AND '2023-01-01' BETWEEN PI.VALID_FROM_DATE AND PI.VALID_TO_DATE QUALIFY ROW_NUMBER() OVER (PARTITION BY POLICY_AGREEMENT_ID  ORDER BY PI.VALID_FROM_DATE DESC,PI.PARTY_ID DESC,PI.INSURANCE_ROLE_PARTY_ID DESC) = 1"

#create teradata connection

uda = teradata.UdaExec(appName='project_name', version='1.0')
connection = uda.connect(driver='Teradata', 
                 method='odbc', system='TDPROD', username=os.environ.get('user'), 
                 password=getpass.getpass(), authentication="LDAP")

df = pd.read_sql(sql,connection)
df = df.groupby(by = ['PARTY_ID']).first()

# --- Code Block ---
import surgeo
fsg = surgeo.BIFSGModel()
sg = surgeo.SurgeoModel()

fsg_results = fsg.get_probabilities(test['FIRST_NAME'], test['LAST_NAME'], test['POST_CODE'])
#sg_results = sg.get_probabilities(test['LAST_NAME'], test['POST_CODE'])

# Fill any null values in fsg_results with values from sg_results
#fsg_results = fsg_results.fillna(sg_results)

# Display the combined results
fsg_results

# --- Code Block ---
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['race_eth'], df['threshold'])

chi2, pvalue, dof, expected = chi2_contingency(contingency_table, correction=False)
print(f'Observed chi2: {chi2:.9f}')
print(f'p-value: {pvalue:.9f}')

# --- Code Block ---
from itertools import combinations
races = ['white', 'api', 'black', 'hispanic']

# List to store results
results = []

# Perform pairwise comparison
for combo in combinations(races, 2):
    # Select the rows for the pair of races
    sub_table = contingency_table.loc[list(combo), :]
    
    # Perform Chi-square test
    chi2, pvalue, dof, expected = chi2_contingency(sub_table, correction=False)
    
    # Append results to the list
    results.append({
        'Comparison': f'{combo[0]} vs {combo[1]}',
        'Chi2': chi2,
        'p-value':  round(pvalue, 5),
        'Degrees of Freedom': dof
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
results_df

# --- Code Block ---
import statsmodels.api as sm 
df_control['Intercept'] = 1
X_control = df_control.drop(columns = ['threshold'])
X_control = X_control.loc[:, (X_control != X_control.iloc[0]).any()] #remove columns that have zero variance
y_control = df['threshold']

log_reg = sm.Logit(y_control, X_control).fit() 
log_reg.summary()

# --- Code Block ---
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Separate features and target
X = df.drop(columns='threshold')
y = df['threshold']

# Define column types for preprocessing
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features_with_race = X.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features_without_race = [col for col in categorical_features_with_race if col != 'race_eth']

# Preprocessor with 'race_eth' column included
preprocessor_with_race = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', dtype=int), categorical_features_with_race)
    ]
)

# Preprocessor without 'race_eth' column
preprocessor_without_race = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', dtype=int), categorical_features_without_race)
    ]
)

# Pipeline with 'race_eth' included
pipeline_with_race = Pipeline([
    ('preprocessor', preprocessor_with_race),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Pipeline without 'race_eth'
pipeline_without_race = Pipeline([
    ('preprocessor', preprocessor_without_race),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train both models
pipeline_with_race.fit(X, y)
pipeline_without_race.fit(X.drop(columns=['race_eth']), y)

# --- Code Block ---
#package import
import numpy as np
import pandas as pd
import os
import getpass
import teradata
import dask.dataframe as dd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotly.express as px
import json
import re


#get pass
os.getcwd()

#set project directory
path = r"\\file018\PMO_GRPB\RENEWAL\CMS Data\SHDS\AC Replacement Model"

# --- Code Block ---
#get all the feature names - queries to get distinct categories form specified tables
ccsr = pd.read_sql("SELECT DISTINCT CCSR_CATEGORY_1 AS CCSR_CATEGORY FROM P_CXS_SBXD.DXCCSR_v2024 ORDER BY CCSR_CATEGORY_1", connection)
ccs = pd.read_sql("SELECT DISTINCT CCS FROM P_CXS_SBXD.CCS_Procedures_v2022 ORDER BY CCS", connection)
drgs = pd.read_sql("SELECT DISTINCT RIGHT(DIAGNOSIS_RELATED_GROUP_INFORMATION,3) DRG FROM P_SDM_ETRAM_STG_VW.V_ETRAM_INSTITUTIONAL_CLAIM ORDER BY DRG WHERE DRG <> 'None'", connection)

# --- Code Block ---
#stardate
from datetime import datetime
startdate = datetime.now().replace(year=datetime.now().year - 1).strftime('%Y-%m-%d')
#startdate = '2023-10-01'

#policies
pol_sql = open(path + "\SQL\Prod_Policies.sql",'r').read()
pol_sql = pol_sql.format(StartDate= startdate)
pol = pd.read_sql(pol_sql,connection)
pol.columns = pol.columns.str.lower()

# --- Code Block ---
import re
with open('features.txt', 'r') as file:
    feas = [line.strip() for line in file.readlines()]
    
# Fix column names in feas and X_train
feas = [re.sub(r'\.0', '_0', col) for col in feas]
pol.columns = [re.sub(r'\.0', '_0', col) for col in pol.columns]

for col in feas:
    if col not in pol.columns:
        pol[col] = 0 
df = pol[feas]
df.head(3)

# --- Code Block ---
import pickle
with open('lasso_model_refit_07282025.pkl', 'rb') as file:
    ls_model = pickle.load(file)

y_ls_pred = ls_model.predict(df)
df['y_pred'] = y_ls_pred

# --- Code Block ---
#package import
import numpy as np
import pandas as pd
import os
import getpass
#import teradata
import dask.dataframe as dd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotly.express as px
import json
from scipy.sparse import csr_matrix
import pickle

#get pass
os.getcwd()

#set project directory
#path = r"\\file018\PMO_GRPB\RENEWAL\CMS Data\SHDS\AC Replacement Model"

# --- Code Block ---
import lightgbm as lgb
print(lgb.__version__)
!pip show teradata

# --- Code Block ---
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pickle


# Load Lasso parameters
ls = pd.read_csv('Lasso_all_feas_06.26.csv')
best_alpha = ls.sort_values(by='value')['params_alpha'].values[0]    

#training data
y_train = X_train['a_to_e_value']
X_train = X_train.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])


        
#Define numeric and categorial features
numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

#Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
categorical_transformer = OneHotEncoder(drop='first', 
                                        #sparse=True, 
                                        handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

model = Lasso(alpha=best_alpha)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model
pipeline.fit(X_train, y_train)

# --- Code Block ---
#retrain 

from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
from scipy.sparse import csr_matrix


        
#Define numeric and categorial features
numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

#Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
categorical_transformer = OneHotEncoder(drop='first', 
                                        #sparse=True, 
                                        handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

# Initialize LightGBM model
model = LGBMRegressor(**best_params)

lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model
lgbm_pipeline.fit(X_train, y_train)

# --- Code Block ---
import seaborn as sns
sns.histplot(y_test, kde=True)

# --- Code Block ---
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

print(f"lasso R2: ", r2_score(y_test, y_ls_pred))
print(f"lasso RMSE: ", sqrt(mean_squared_error(y_test, y_ls_pred)))
print(f"lasso ME: ", np.mean(y_ls_pred - y_test))
print("\n")
print(f"lgbm R2: ", r2_score(y_test, y_pred))
print(f"lgbm RMSE: ", sqrt(mean_squared_error(y_test, y_pred)))
print(f"lgbmo ME: ", np.mean(y_pred - y_test))

# --- Code Block ---
import plotly.express as px

# Creating the DataFrame for plotting
data = pd.DataFrame({
    'Measurement': ['y_test', 'y_lasso', 'y_lgbm'],
    'Average': [y_test.mean(), y_ls_pred.mean(), y_pred.mean()]
})

# Plot using Plotly Express
fig = px.bar(data, x='Measurement', y='Average', text='Average',
                 title='Model - Comparison of Averages: y_test vs. y_lasso vs. y_lgbm',
                 labels={'Measurement': 'Variable', 'Average': 'Average Value'})

# Show the plot
fig.show()

# --- Code Block ---
# Generate thresholds with a step of 0.1
thresholds = [round(x * 0.1, 1) for x in range(0, 21)]  # This covers from 0.0 to 2.0
def ae_group(average_a_to_e):
    # Find the right interval for the average_a_to_e
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= average_a_to_e < thresholds[i+1]:
            return f'{thresholds[i]} - {thresholds[i+1]}'
    # Handle the case where the value is greater than or equal to the last threshold
    return f'{thresholds[-1]}+'

# --- Code Block ---
averages = decile.groupby('ae_group').mean().reset_index()
group_counts = decile.groupby('ae_group').size().reset_index(name='count')


# Creating the plot with dual-axis
import plotly.graph_objects as go
fig = go.Figure()

# Adding line charts for averages
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_test'], name='Average y_test',
                         mode='lines+markers', yaxis='y2', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_lasso'], name='Average y_lasso',
                         mode='lines+markers', yaxis='y2', line=dict(color='red')))
fig.add_trace(go.Scatter(x=averages['ae_group'], y=averages['y_lgbm'], name='Average y_lgbm',
                         mode='lines+markers', yaxis='y2', line=dict(color='green')))

# Adding bar chart for counts
fig.add_trace(go.Bar(x=group_counts['ae_group'], y=group_counts['count'], name='Count',
                     marker=dict(color='grey'), yaxis='y1'))

# Layout with secondary y-axis for line charts
fig.update_layout(
    title='Average Values and Counts per AE Group',
    xaxis_title='AE Group',
    yaxis=dict(title='Count', side='right'),
    yaxis2=dict(title='Average Value', overlaying='y', side='left'),
    legend_title='Metrics',
    template='plotly_white'
)

# Show the figure
fig.show()

# --- Code Block ---
import seaborn as sns
import matplotlib.pyplot as plt

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(decile[['y_test','y_lasso','y_lgbm']].corr(), annot=True, cmap='crest', linewidths=0.5, linecolor='black', fmt=".2f", square=True)
plt.title('Heatmap of Correlation Matrix')

# Show the plot
plt.show()

# --- Code Block ---
# Creating the Plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lasso, mode='lines+markers',
                         name='Cumulative Acceptance Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lgbm, mode='lines+markers',
                         name='Cumulative Acceptance LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_true, mode='lines+markers',
                         name='Cumulative Acceptance True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))
fig.update_layout(
    title='Cumulative Acceptance for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Cumulative Acceptance',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
# Creating the Plotly graph
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lasso_percent, mode='lines+markers',
                         name='Cumulative Acceptance Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_lgbm_percent, mode='lines+markers',
                         name='Cumulative Acceptance LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_accepts_true_percent, mode='lines+markers',
                         name='Cumulative Acceptance True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))
fig.update_layout(
    title='Cumulative Acceptance (%) for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Cumulative Acceptance(%)',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
import plotly.graph_objects as go
thresholds = np.linspace(0, 2, 300)

# Calculating cumulative sums for each threshold for each model
cumulative_sums_lasso = [decile[decile['y_lasso'] < t]['y_test'].mean() for t in thresholds]
cumulative_sums_lgbm = [decile[decile['y_lgbm'] < t]['y_test'].mean() for t in thresholds]
#cumulative_sums_true = [decile[decile['y_test'] < t]['y_test'].mean() for t in thresholds]

# Creating the Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
'''fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_true, mode='lines+markers',
                         name='Cumulative Mean True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))'''
fig.update_layout(
    title='Mean a_to_e Values for Lasso vs. LGBM Below Thresholds',
    xaxis_title='Threshold Value',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import re


# Load Lasso parameters
ls = pd.read_csv('Lasso_all_feas_06.26.csv')
best_alpha = ls.sort_values(by='value')['params_alpha'].values[0] 

# Fix column names in feas and X_train
feas = [re.sub(r'\.0', '_0', col) for col in feas]
X_train.columns = [re.sub(r'\.0', '_0', col) for col in X_train.columns]

#training data
y_train = X_train['a_to_e_value']
for col in feas:
    if col not in X_train.columns:
        X_train[col] = 0 
X_train = X_train[feas]


        
#Define numeric and categorial features
numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

#Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
categorical_transformer = OneHotEncoder(drop='first', 
                                        #sparse=True, 
                                        handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough', sparse_threshold = 1) #ensure the output is sparse

model = Lasso(alpha=best_alpha)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model
pipeline.fit(X_train, y_train)

# --- Code Block ---
import sklearn
print(sklearn.__version__)

# --- Code Block ---
import re
with open('features.txt', 'r') as file:
    feas = [line.strip() for line in file.readlines()]
    
# Fix column names in feas and X_train
feas = [re.sub(r'\.0', '_0', col) for col in feas]
test_df.columns = [re.sub(r'\.0', '_0', col) for col in test_df.columns]

for col in feas:
    if col not in test_df.columns:
        test_df[col] = 0 
test_df = test_df[feas]

# --- Code Block ---
import pickle
with open('lasso_model_refit_07282025.pkl', 'rb') as file:
    ls_model = pickle.load(file)

y_ls_pred = ls_model.predict(df)
df['y_pred'] = y_ls_pred

