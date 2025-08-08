# --- Code Block ---
#Pandas Dataframe
# Get a list of all feather files in the directory
'''files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('.feather')]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each feather file, read it, and append to the list
for file in files:
    filepath = os.path.join(path + '\Data\\', file)
    df = pd.read_feather(filepath)
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
dfs = pd.concat(dfs, ignore_index=True)'''

# --- Code Block ---
# Dask Dataframe
'''files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('.feather')]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each feather file, read it, and append to the list
for file in files:
    filepath = os.path.join(path + '\Data\\', file)
    df = feather.read_feather(filepath)
    #df.fillna(0, inplace=True)
    #for col in df.select_dtypes(include=['float', 'int']).columns:
        #df[col] = pd.to_numeric(df[col], downcast='integer')
        
    df = dd.from_pandas(df, npartitions=1)  # Convert to Dask DataFrame
    
    dfs.append(df)

# Concatenate the Dask DataFrames into a single Dask DataFrame
dfs = dd.concat(dfs, axis=0)
dfs = dfs.drop(columns=['policy_agreement_id'])'''

# --- Code Block ---
ae_stats = pd.concat(ae_stats, ignore_index=True)
ae_stats

# --- Code Block ---
#states averages
files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('.feather')]

# Initialize an empty list to store DataFrames
states = []

# Loop through each feather file, read it, and append to the list
for file in files:
    filepath = os.path.join(path + '\Data\\', file)
    z = pd.read_feather(filepath,columns=['state', 'a_to_e_value'])
    
    states.append(z)

#Concatenate all DataFrames into a single DataFrame
states = pd.concat(states, ignore_index=True)

states = states.groupby('state')[['a_to_e_value']].mean().reset_index()
states.to_csv(path + '\Data\\states.csv')

# --- Code Block ---
all_correlations = pd.concat(all_correlations, axis=1)
all_correlations.to_csv('all_correlations.csv')

# --- Code Block ---
#rename df columns
ccs_cols = pd.DataFrame(df.filter(regex='^(ccs_)').columns, columns = ['feature'])
ccs_cols["ccs|ccsr"] = ccs_cols["feature"].apply(lambda x: x.split("_")[-1])

ccsr_cols = pd.DataFrame(df.filter(regex='^(ccsr_)').columns, columns = ['feature'])
ccsr_cols["ccs|ccsr"] = ccsr_cols["feature"].apply(lambda x: x.split("_")[-1])

ccs_cols = pd.merge(ccs_cols, ccs_label, left_on='ccs|ccsr', right_on='CCS', how='left')
ccs_cols["label"] = ccs_cols["feature"] + " - " + ccs_cols["CCS Label"]
ccsr_cols = pd.merge(ccsr_cols, ccsr_label, left_on='ccs|ccsr', right_on='CCSR_CATEGORY_1', how='left')
ccsr_cols["label"] = ccsr_cols["feature"] + " - " + ccsr_cols["CCSR_CATEGORY"]
ccs_cols = ccs_cols[['feature','label']]
ccsr_cols = ccsr_cols[['feature','label']]

df1 = pd.concat([ccs_cols,ccsr_cols])

for index, row in df1.iterrows():
    feature = row['feature']
    label = row['label']
    if feature in df.columns:
        df.rename(columns={feature: label}, inplace=True)

# --- Code Block ---
#Clean the data folds for training

# Get a list of all feather files in the directory
files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('01.feather')]

#zip mapping
file_path = 'zip_to_category.json'
with open(file_path, 'r') as f:
    zip_to_category = json.load(f)

#drop_features (feature with 100% missing values and zero/null correlations - see eda notebook ) 
with open('drop_features.txt', 'r') as file:
    nan_features = [line.strip() for line in file]

#Abs Pearson correlations < 0.01
corr = pd.read_csv('all_correlations.csv')
corr.rename(columns = {'Unnamed: 0':'feature'}, inplace = True)
corr = corr[corr.isna().any(axis=1)]
corr_features = corr['feature'].tolist()

#las = pd.read_csv('lasso_coefs.csv')
#las_features = las[las['Coefficient'] == 0]['Feature'].tolist()

drop_features = nan_features + corr_features  
#las_features

#read the feather file
for i, file in enumerate(files, start=1):
    filepath = os.path.join(path + '\Data\\', file)
    df = pd.read_feather(filepath)
        
    #Apply the mapping to the 'zip' column 
    df['zip_category'] = df['zip'].map(zip_to_category)

     #downcast float and int column
    for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
    #Clean object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].astype(str)

    df = df[df['plan'].isin(['F', 'G', 'N'])]
    df = df.drop(columns=drop_features, errors = 'ignore')
    df.drop(columns=['zip', 'startdate'],inplace=True)
    df.fillna(0, inplace=True)

    new_file_name = f"fold{i}.feather"  # Define the new file name
    df.reset_index(drop=False).to_feather(os.path.join(path + '\Data\\' + new_file_name))

# --- Code Block ---
results_df.to_csv('Lasso_all_feas_06.20.csv')

# --- Code Block ---
results_df.to_csv('lasso_etram_04.19.csv')

# --- Code Block ---
results_df.to_csv('gbm_all_feas_04.19.csv')

# --- Code Block ---
uw = pd.read_csv(os.path.join(path + '\Data\\UW_Results_All.csv'))
uw.describe()

# --- Code Block ---
test = pd.read_feather(os.path.join(path + '\Data\\fold5.feather'))
ae = test
ae['y_test'] = y_test
ae['y_pred'] = y_pred
ae['y_lasso'] = y_ls_pred
ae = ae[['party_id','y_test','y_pred','y_lasso']]
ae = pd.merge(ae, uw, left_on='party_id',right_on='PARTY_ID', how = 'inner')

# --- Code Block ---
sp = ae[['y_test','y_pred','y_lasso','CLAIMS_LOSS_RATIO_SCORE_NUM','Predicted A_TO_E']]
sp.rename(columns={'Predicted A_TO_E': 'y_ae_affiliate_conversion'}, inplace=True)
sp.to_csv('models_scores.csv')

# --- Code Block ---
results_df = study4.trials_dataframe()

# Calculate and add average metrics to the dataframe
for metric in ['mse', 'rmse', 'r2']:
    train_metrics = results_df.filter(like=f'train_{metric}').mean(axis=1)
    test_metrics = results_df.filter(like=f'test_{metric}').mean(axis=1)
    results_df[f'average_train_{metric}'] = train_metrics
    results_df[f'average_test_{metric}'] = test_metrics

results_df.to_csv('gmb_etram_04.19.csv')

# --- Code Block ---
#Use this for expanding window
'''for i in range(len(file_names) - window_size + 1):
        train_dfs = [pd.read_feather(os.path.join(directory_path, file_names[j])) for j in range(i, i + window_size - 1)]
        combined_train_df = pd.concat(train_dfs)
        test_df = pd.read_feather(os.path.join(directory_path, file_names[i + window_size - 1]))'''

# --- Code Block ---
X_train = []
file_names = [f for f in os.listdir() if f.startswith('fold')]
file_names.sort()

for file in file_names[0:5]:
    df = pd.read_feather(file)
    X_train.append(df)

X_train = pd.concat(X_train, ignore_index=True)
X_train.info()

# --- Code Block ---
uw = pd.read_csv('UW_Results_All.csv')
uw.describe()

# --- Code Block ---
test = pd.read_feather('fold5.feather')
ae = test
ae['y_test'] = y_test
ae['y_pred'] = y_pred
ae['y_lasso'] = y_ls_pred
ae = ae[['party_id','y_test','y_pred','y_lasso']]
ae = pd.merge(ae, uw, left_on='party_id',right_on='PARTY_ID', how = 'inner')

# --- Code Block ---
sp = ae[['party_id','y_test','y_pred','y_lasso','CLAIMS_LOSS_RATIO_SCORE_NUM','Predicted A_TO_E']]
sp.rename(columns={'Predicted A_TO_E': 'y_ae_affiliate_conversion'}, inplace=True)
sp.to_csv('models_scores.csv')

# --- Code Block ---
test = pd.merge(test_df, df, left_on='party_id',right_on='PARTY_ID', how = 'left')
test.info()

# --- Code Block ---
df = pd.concat([test, fsg_results['race_eth']], axis=1)

# --- Code Block ---
#categorial columns
categorical_columns = df_control.select_dtypes(include=['object', 'category']).columns

#numeric columns
numeric_df = df_control.select_dtypes(include=['number'])

#Create dummy variables for race_eth excluding 'native'
race_eth_dummies = pd.get_dummies(df_control['race_eth'], dtype=int)
race_eth_dummies = race_eth_dummies.drop(columns=['native'], errors='ignore')

# Create dummy variables for other categorical columns with drop_first=True
other_categorical_columns = categorical_columns.drop('race_eth')
other_dummies_df = pd.get_dummies(df_control[other_categorical_columns], drop_first=True, dtype=int)

#Concatenate numeric, race_eth dummies, and other dummy variables
df_control = pd.concat([numeric_df, race_eth_dummies, other_dummies_df], axis=1)

# --- Code Block ---
# Extract coefficients
coefficients_with_race = pipeline_with_race.named_steps['classifier'].coef_[0]
coefficients_without_race = pipeline_without_race.named_steps['classifier'].coef_[0]

# Create a DataFrame to compare coefficients
feature_names_with_race = pipeline_with_race.named_steps['preprocessor'].get_feature_names_out()
feature_names_without_race = pipeline_without_race.named_steps['preprocessor'].get_feature_names_out()

# Build DataFrames for coefficients
df_coefficients_with_race = pd.DataFrame({
    'Feature': feature_names_with_race,
    'Coefficient (With race_eth)': coefficients_with_race
})

df_coefficients_without_race = pd.DataFrame({
    'Feature': feature_names_without_race,
    'Coefficient (Without race_eth)': coefficients_without_race
})

# Merge the two DataFrames for easy comparison
df_comparison = pd.merge(df_coefficients_with_race, df_coefficients_without_race, on='Feature', how='outer')

# Display the comparison
df_comparison

# --- Code Block ---
# Initialize an empty list to store the results for each race_eth group
results = []

# Loop through each unique value in the 'race_eth' column
for race in [ 'white', 'hispanic', 'black', 'native', 'api']:
    # Filter the DataFrame for the current race_eth value
    group_df = df[df['race_eth'] == race]
    
    # Calculate the percentage of non-zeros for each column (excluding 'race_eth')
    percentage_non_zero = (group_df.iloc[:, 1:].astype(bool).sum(axis=0) / len(group_df)) * 100
    
    # Create a DataFrame for the results
    result_df = percentage_non_zero.reset_index()
    result_df.columns = ['Column', 'Percentage_Non_Zero']
    result_df['race_eth'] = race  # Add the race_eth value
    
    # Append the result to the list
    results.append(result_df)

# Combine all results into a single DataFrame
final_results = pd.concat(results, ignore_index=True)

# --- Code Block ---
final_results.to_csv('Disease Prevalence.csv')

# --- Code Block ---
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

# --- Code Block ---
#pol.to_csv("predictions.csv", index=False)

# --- Code Block ---
X_train = []
file_names = [f for f in os.listdir() if f.startswith('fold')]
file_names.sort()

for file in file_names[0:5]:
    df = pd.read_feather(file)
    X_train.append(df)

X_train = pd.concat(X_train, ignore_index=True)
X_train.info()

# --- Code Block ---
X_train = []
file_names = [f for f in os.listdir() if f.startswith('fold')]
file_names.sort()

for file in file_names:
    df = pd.read_feather(file)
    X_train.append(df)

X_train = pd.concat(X_train, ignore_index=True)
X_train.info()

