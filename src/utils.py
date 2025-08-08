# --- Code Block ---
ccs_label['CCS'] = ccs_label['CCS'].astype('str')
ccsr_label["CCSR_CATEGORY_1"] = ccsr_label["CCSR_CATEGORY_1"].str.lower()

# --- Code Block ---
# Filter columns for correlation calculation
correlation_columns = [col for col in df.columns if col.startswith('ccsr_3mos')] + ['a_to_e_value']

# Calculate correlation matrix
correlation_matrix = df[correlation_columns].corrwith(df['a_to_e_value'])
correlation_matrix.drop('a_to_e_value').abs().sort_values(ascending=False).head(10)

# --- Code Block ---
# Select top 10 ccsr_3mos correlated columns 
top_10_ccsr_3mos = correlation_matrix.drop('a_to_e_value').abs().sort_values(ascending=False).head(10).index.tolist()

# --- Code Block ---
df['high_ae'] = df['a_to_e_value'].apply(lambda x: 1 if x >=10 else 0)
df_ae = df[['high_ae','claims_3mos', 'claims_6mos','claims_12mos']].groupby(by=['high_ae']).agg({'claims_3mos':['mean'],'claims_6mos':['mean'],'claims_12mos':['mean']})
df_ae.columns = df_ae.columns.levels[0]
#for col in df_ae.columns:
    #df_ae[col] = df_ae[col].map(lambda x: f"${x/1000000:.1f}M")
df_ae

# --- Code Block ---


# --- Code Block ---
plot_slice(study)

# --- Code Block ---
plot_edf(study)

# --- Code Block ---
results_df.head(1)

# --- Code Block ---
#test set
# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('fold')]
directory_path = os.path.join(path + '\Data\\')

test_df = pd.read_feather(directory_path + file_names[4])
X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
y_test = test_df['a_to_e_value']

# --- Code Block ---
with open('trained_pipeline_lasso_final.pkl', 'rb') as file:
    ls_model = pickle.load(file)

# --- Code Block ---
decile = pd.DataFrame({'y_test':y_test,'y_ls_pred':y_ls_pred})
decile['ae_group'] = decile['y_test'].apply(ae_group)

# --- Code Block ---
averages = decile.groupby('ae_group').mean()
averages.reset_index(inplace=True)

fig = px.bar(averages, x='ae_group', y=['y_test', 'y_ls_pred'], 
             title='Average of y_test and y_ls_pred per group',
             labels={'value': 'Average Value', 'variable': 'Variables', 'ae_group': 'ae_group'},
             barmode='group')

# Show the figure
fig.show()

# --- Code Block ---
# Creating the line chart
fig = px.line(averages, x='ae_group', y=['y_test', 'y_ls_pred'], 
              title='Average of y_test and y_ls_pred per AE Group',
              labels={'value': 'Average Value', 'variable': 'Variables', 'ae_group': 'AE Group'},
              markers=True)

# Show the figure
fig.show()

# --- Code Block ---
with open('trained_pipeline_lgbm.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

#test set
test_df = pd.read_feather(directory_path + file_names[4])
X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','zip_category','a_to_e_value'])
y_test = test_df['a_to_e_value']

# --- Code Block ---
sns.histplot(y_test, kde=True)

# --- Code Block ---
sns.histplot(y_pred, kde=True)

# --- Code Block ---
print(r2_score(y_test, y_pred))
print(sqrt(mean_squared_error(y_test, y_pred)))
print(np.mean(y_pred - y_test))

# --- Code Block ---
data.mean()

# --- Code Block ---
decile = pd.DataFrame({'y_test':y_test,'y_lasso': y_ls_pred, 'y_lgbm':y_pred})
decile['ae_group'] = decile['y_test'].apply(ae_group)

# --- Code Block ---
averages = decile.groupby('ae_group').mean()
averages.reset_index(inplace=True)

fig = px.bar(averages, x='ae_group', y=['y_test','y_lasso', 'y_lgbm'], 
             title='Average of a_to_e per group',
             labels={'value': 'Average Value', 'variable': 'Variables', 'ae_group': 'ae_group'},
             barmode='group')

# Show the figure
fig.show()
#bin 2+, keep y_test as bar, others as lines

# --- Code Block ---
sns.histplot(uw['A_TO_E_VALUE'])

# --- Code Block ---
uw['Predicted A_TO_E'] = y_uw

# --- Code Block ---
print(r2_score(y, y_uw))
print(sqrt(mean_squared_error(y, y_uw)))

# --- Code Block ---
sp

# --- Code Block ---
'''
# Define file names and directory path
file_names = [f for f in os.listdir(path + '\Data\\' ) if f.startswith('df')]
directory_path = os.path.join(path + '\Data\\')

# Define the sliding window size for training
window_size = 1

for i in range(len(file_names)- 1):
        # Load dataframes for the current training window
        print(directory_path + file_names[i])#training
        print(directory_path + file_names[i+1])#testing'''

# --- Code Block ---
df.head(5)

# --- Code Block ---
df.filter(regex='^(ccs|ccsr|drg)')

# --- Code Block ---
print(pyarrow.__version__)
print(pd.__version__)

# --- Code Block ---
test_df = pd.read_feather(file_names[5])
test_df.info()

# --- Code Block ---
# Save the trained pipeline to a file
with open('lasso_model.pkl', 'wb') as file:
            pickle.dump(pipeline, file)

# --- Code Block ---
# Save the trained pipeline to a file
with open('lgbm_model.pkl', 'wb') as file:
            pickle.dump(lgbm_pipeline, file)

# --- Code Block ---
#Load Lasso vs Light-GBM
with open('lgbm_model.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

with open('lasso_model.pkl', 'rb') as file:
    ls_model = pickle.load(file)

# --- Code Block ---
#test set
test_df = pd.read_feather("fold5.feather")
X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
y_test = test_df['a_to_e_value']

# --- Code Block ---
sns.histplot(y_ls_pred, kde=True)

# --- Code Block ---
sns.histplot(y_pred, kde=True)

# --- Code Block ---
data

# --- Code Block ---
decile = pd.DataFrame({'y_test':y_test,'y_lasso': y_ls_pred, 'y_lgbm':y_pred})
decile['ae_group'] = decile['y_test'].apply(ae_group)

# --- Code Block ---
averages = decile.groupby('ae_group').mean()
averages.reset_index(inplace=True)

fig = px.bar(averages, x='ae_group', y=['y_test','y_lasso', 'y_lgbm'], 
             title='Average of a_to_e per group',
             labels={'value': 'Average Value', 'variable': 'Variables', 'ae_group': 'ae_group'},
             barmode='group')

# Show the figure
fig.show()

# --- Code Block ---
sns.histplot(uw['A_TO_E_VALUE'])

# --- Code Block ---
uw['Predicted A_TO_E'] = y_uw

# --- Code Block ---
print(r2_score(y, y_uw))
print(sqrt(mean_squared_error(y, y_uw)))

# --- Code Block ---
sp

# --- Code Block ---
#test set
test_df = pd.read_feather(directory_path + 'fold5.feather')
X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
y_test = test_df['a_to_e_value']

# --- Code Block ---
fsg_results['race_eth'] = fsg_results[['white', 'black', 'api', 'native', 'multiple', 'hispanic']].idxmax(axis=1)

# --- Code Block ---
df['MARITAL_STATUS_NME'] = df['MARITAL_STATUS_NME'].apply(lambda x: np.nan if x == 'Unknown' else x)

# --- Code Block ---
df.isna().sum()[df.isna().sum() != 0]

# --- Code Block ---
df['race_eth'].isna().mean()*100

# --- Code Block ---
df['MARITAL_STATUS_NME'].isna().mean()*100

# --- Code Block ---
df.groupby(by = ['MARITAL_STATUS_NME'])['MARITAL_STATUS_NME'].count()

# --- Code Block ---
# Count the occurrences of each category in 'race_eth'
race_eth_counts = df['race_eth'].value_counts().reset_index()

# Calculate the percentage for each category
race_eth_counts['percentage'] = (race_eth_counts['count'] / race_eth_counts['count'].sum()) * 100

race_eth_counts

# --- Code Block ---
sex_counts = df['gender'].value_counts().reset_index()

# Calculate the percentage for each category
sex_counts['percentage'] = (sex_counts['count'] / sex_counts['count'].sum()) * 100

sex_counts

# --- Code Block ---
df['gender'].isna().mean()*100

# --- Code Block ---
counts = df['MARITAL_STATUS_NME'].value_counts().reset_index()

# Calculate the percentage for each category
counts['percentage'] = (counts['count'] / counts['count'].sum()) * 100

counts

# --- Code Block ---
df['MARITAL_STATUS_NME'].isna().mean()*100

# --- Code Block ---
df.columns

# --- Code Block ---
df_control.info()

# --- Code Block ---
df = df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value','y_pred','FIRST_NAME','LAST_NAME','POST_CODE','MARITAL_STATUS_NME'])

# --- Code Block ---
df_comparison['Coef_Diff'] = np.abs(df_comparison['Coefficient (With race_eth)'] - df_comparison['Coefficient (Without race_eth)'])
df_comparison[df_comparison['Coef_Diff'] == 0]

# --- Code Block ---
df_comparison.shape

# --- Code Block ---
df_comparison['Coef_Diff'].hist(bins=50)

# --- Code Block ---
df_comparison.sort_values(by = 'Coef_Diff', ascending=False).head(30)

# --- Code Block ---
df.columns

# --- Code Block ---
df.groupby(by = ['race_eth'])[['claims_3mos','claims_6mos','claims_12mos','claims_2yrs','claims_3yrs']].mean().transpose()

# --- Code Block ---
# Calculate the percentage of non-zeros for each column
percentage_non_zero = (df.astype(bool).sum(axis=0) / len(df)) * 100

# Display the results as a DataFrame
result_df = percentage_non_zero.reset_index()
result_df.columns = ['Column', 'Percentage_Non_Zero']

# --- Code Block ---
result_df

# --- Code Block ---
final_results[final_results['Column'].str.startswith('ccsr_12mos')].sort_values(by='Percentage_Non_Zero')

# --- Code Block ---
pol.info()

# --- Code Block ---
icd.head(2)

# --- Code Block ---
cpt.head(3)

# --- Code Block ---
df[['y_pred']].describe()

# --- Code Block ---
# Plot the histogram
plt.figure(figsize=(20, 6))
plt.hist(df['y_pred'], bins=1000, color='blue', edgecolor='black')
plt.xlabel('y_pred')
plt.ylabel('Frequency')
plt.title('Histogram of Value Column')
# Add a vertical line at x = 1
plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
#plt.axvline(x=.75, color='red', linestyle='--', linewidth=2)
plt.show()

# --- Code Block ---
pol['y_pred'] = y_ls_pred

# --- Code Block ---
#pol[pol['party_id'] == 553446408].transpose()

# --- Code Block ---
#use these packages for importing feather files
!pip install pandas==2.2.1 pyarrow==15.0.2 --upgrade

# --- Code Block ---
test_df = pd.read_feather(file_names[5])
test_df.info()

# --- Code Block ---
# Save the trained pipeline to a file
with open('lasso_model_032025.pkl', 'wb') as file:
            pickle.dump(pipeline, file)

# --- Code Block ---
# Save the trained pipeline to a file
with open('lgbm_model_032025.pkl', 'wb') as file:
            pickle.dump(lgbm_pipeline, file)

# --- Code Block ---
#Load Lasso vs Light-GBM
with open('lgbm_model_032025.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

with open('lasso_model_032025.pkl', 'rb') as file:
    ls_model = pickle.load(file)

# --- Code Block ---
#test set
test_df = pd.read_feather("fold6.feather")
X_test = test_df.drop(columns=['index','party_id','policy_agreement_id','gender','claims_mm','zip_category','a_to_e_value'])
y_test = test_df['a_to_e_value']

# --- Code Block ---
sns.histplot(y_ls_pred, kde=True)

# --- Code Block ---
sns.histplot(y_pred, kde=True)

# --- Code Block ---
data

# --- Code Block ---
decile = pd.DataFrame({'y_test':y_test,'y_lasso': y_ls_pred, 'y_lgbm':y_pred})
decile['ae_group'] = decile['y_test'].apply(ae_group)

# --- Code Block ---
averages = decile.groupby('ae_group').mean()
averages.reset_index(inplace=True)

fig = px.bar(averages, x='ae_group', y=['y_test','y_lasso', 'y_lgbm'], 
             title='Average of a_to_e per group',
             labels={'value': 'Average Value', 'variable': 'Variables', 'ae_group': 'ae_group'},
             barmode='group')

# Show the figure
fig.show()

# --- Code Block ---
X_train = []
file_names = [f for f in os.listdir() if f.startswith('fold')]
file_names.sort()

# --- Code Block ---
test_df = pd.read_feather(file_names[5])
test_df.info()

# --- Code Block ---
df = test_df[feas]
df.head(3)

# --- Code Block ---
df[['y_pred']].describe()

# --- Code Block ---
# Plot the histogram
plt.figure(figsize=(20, 6))
plt.hist(df['y_pred'], bins=1000, color='blue', edgecolor='black')
plt.xlabel('y_pred')
plt.ylabel('Frequency')
plt.title('Histogram of Value Column')
# Add a vertical line at x = 1
plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
#plt.axvline(x=.75, color='red', linestyle='--', linewidth=2)
plt.show()

