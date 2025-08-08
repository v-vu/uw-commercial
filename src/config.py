# --- Code Block ---
#Create a dictionary for zip category
files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('.feather')]

# Initialize an empty list to store DataFrames
zips = []

# Loop through each feather file, read it, and append to the list
for file in files:
    filepath = os.path.join(path + '\Data\\', file)
    z = pd.read_feather(filepath,columns=['zip', 'a_to_e_value'])
    
    zips.append(z)

#Concatenate all DataFrames into a single DataFrame
zips = pd.concat(zips, ignore_index=True)


#average 'a_to_e_value' for each zip code
average_a_to_e_by_zip = zips.groupby('zip')['a_to_e_value'].mean()
thresholds = [0.5, 0.75, 1, 1.5, 2, 5, 10]

# define function to map each average 'a_to_e_value' to its corresponding category
def map_to_category(average_a_to_e):
    if average_a_to_e < thresholds[0]:
        return '< 0.5'
    elif thresholds[0] <= average_a_to_e < thresholds[1]:
        return '0.5 - 0.75'
    elif thresholds[1] <= average_a_to_e < thresholds[2]:
        return '0.75 - 1'
    elif thresholds[2] <= average_a_to_e < thresholds[3]:
        return '1 - 1.5'
    elif thresholds[3] <= average_a_to_e < thresholds[4]:
        return '1.5 - 2'
    elif thresholds[4] <= average_a_to_e < thresholds[5]:
        return '2 - 5'
    elif thresholds[5] <= average_a_to_e < thresholds[6]:
        return '5 - 10'
    else:
        return '10+'

#Create a dictionary mapping zip codes to categories based on their average 'a_to_e_value'
zip_to_category = average_a_to_e_by_zip.apply(map_to_category).to_dict()

# Export the dictionary to a JSON file
file_path = 'zip_to_category.json'
with open(file_path, 'w') as f:
    json.dump(zip_to_category, f)

# --- Code Block ---
state_averages = pd.read_csv(path + '\Data\\states.csv')
fig = px.choropleth(state_averages,
                    locations='state',  # DataFrame column with locations
                    locationmode="USA-states",  # Set to match the "locations" parameter
                    color='a_to_e_value',  # DataFrame column with values used for color scale
                    color_continuous_scale="Viridis",  # Color scale
                    scope="usa",  # Focus on the USA
                    labels={'atoe': 'Average ATOE Value'},  # Label for the color scale
                    title='Average ATOE Value by US State')  # Title of the plot

fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))  # Optional: make background transparent

fig.show()

# --- Code Block ---
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 5, 10]

# define function to map each average 'a_to_e_value' to its corresponding category
def ae_group(average_a_to_e):
    if average_a_to_e < thresholds[0]:
        return '0.0 - 0.2'
    elif thresholds[0] <= average_a_to_e < thresholds[1]:
        return '0.2 - 0.3'
    elif thresholds[1] <= average_a_to_e < thresholds[2]:
        return '0.3 - 0.4'
    elif thresholds[2] <= average_a_to_e < thresholds[3]:
        return '0.4 - 0.5'
    elif thresholds[3] <= average_a_to_e < thresholds[4]:
        return '0.5 - 0.6'
    elif thresholds[4] <= average_a_to_e < thresholds[5]:
        return '0.6- 0.7'
    elif thresholds[5] <= average_a_to_e < thresholds[6]:
        return '0.7 - 0.8'
    elif thresholds[6] <= average_a_to_e < thresholds[7]:
        return '0.8 - 0.9'
    elif thresholds[7] <= average_a_to_e < thresholds[8]:
        return '0.9- 1'
    elif thresholds[8] <= average_a_to_e < thresholds[9]:
        return '1 - 1.25'
    elif thresholds[9] <= average_a_to_e < thresholds[10]:
        return '1.25 - 1.5'
    elif thresholds[10] <= average_a_to_e < thresholds[11]:
        return '1.5 - 2'
    else:
        return '2+'

# --- Code Block ---
#best hyper-parameters
df_trials = pd.read_csv('gbm_all_feas_04.19.csv')
best_trial = df_trials[df_trials['value'] == df_trials['value'].min()]
best_params = best_trial.filter(regex='params_')  # This regex matches columns that start with 'params_'
best_params.columns = best_params.columns.str.replace('params_', '')
best_params = best_params.iloc[0].to_dict()
for param in ['bagging_freq','max_depth','min_data_in_leaf','num_leaves']:
    best_params[param] = int(best_params[param])
best_params

# --- Code Block ---
thresholds = np.linspace(0, 2, 300)
# Calculating cumulative acceptances for each threshold for both models
cumulative_accepts_lasso = [decile[decile['y_lasso'] < t].shape[0] for t in thresholds]
cumulative_accepts_lgbm = [decile[decile['y_lgbm'] < t].shape[0] for t in thresholds]
cumulative_accepts_true = [decile[decile['y_test'] < t].shape[0] for t in thresholds]

# --- Code Block ---
# Total number of instances
total_instances = decile.shape[0]

# Calculating cumulative acceptances as percentages for each threshold
cumulative_accepts_lasso_percent = [(decile[decile['y_lasso'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_lgbm_percent = [(decile[decile['y_lgbm'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_true_percent = [(decile[decile['y_test'] < t].shape[0] / total_instances) * 100 for t in thresholds]

# --- Code Block ---
# Creating the Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_accepts_lasso_percent, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=cumulative_accepts_lgbm_percent, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
'''fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_true, mode='lines+markers',
                         name='Cumulative Mean True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))'''

fig.update_layout(
    title='Mean a_to_e Values vs Acceptance Rate',
    xaxis_title='Acceptance Rate',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
#calculate acceptance rate vs mean a_to_e value for Affiliate Conversion vs new models
thresholds = np.linspace(0, 2, 300)
total_instances = ae.shape[0]

cumulative_accepts_lasso_percent = [(ae[ae['y_lasso'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_lgbm_percent = [(ae[ae['y_pred'] < t].shape[0] / total_instances) * 100 for t in thresholds]

cumulative_sums_lasso = [ae[ae['y_lasso'] < t]['y_test'].mean() for t in thresholds]
cumulative_sums_lgbm = [ae[ae['y_pred'] < t]['y_test'].mean() for t in thresholds]

# --- Code Block ---
cumulative_accepts_uw = [(ae[ae['Predicted A_TO_E'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_mean_uw = [ae[ae['Predicted A_TO_E'] < t]['y_test'].mean() for t in thresholds]

fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_accepts_lasso_percent, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean - Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=1)))
fig.add_trace(go.Scatter(x=cumulative_accepts_lgbm_percent, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean - LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=1)))
fig.add_trace(go.Scatter(x=cumulative_accepts_uw, y=cumulative_mean_uw,  mode='lines+markers',
                         name='Cumulative Mean - Affilicate Conversion',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=1)))

fig.update_layout(
    title='Mean a_to_e Values vs Acceptance Rate',
    xaxis_title='Acceptance Rate',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
#best hyper-parameters
df_trials = pd.read_csv('gbm_all_feas_06.26.csv')
best_trial = df_trials[df_trials['value'] == df_trials['value'].min()]
best_params = best_trial.filter(regex='params_')  # This regex matches columns that start with 'params_'
best_params.columns = best_params.columns.str.replace('params_', '')
best_params = best_params.iloc[0].to_dict()
for param in ['bagging_freq','max_depth','min_data_in_leaf','num_leaves']:
    best_params[param] = int(best_params[param])
best_params

# --- Code Block ---
thresholds = np.linspace(0, 2, 300)
# Calculating cumulative acceptances for each threshold for both models
cumulative_accepts_lasso = [decile[decile['y_lasso'] < t].shape[0] for t in thresholds]
cumulative_accepts_lgbm = [decile[decile['y_lgbm'] < t].shape[0] for t in thresholds]
cumulative_accepts_true = [decile[decile['y_test'] < t].shape[0] for t in thresholds]

# --- Code Block ---
# Total number of instances
total_instances = decile.shape[0]

# Calculating cumulative acceptances as percentages for each threshold
cumulative_accepts_lasso_percent = [(decile[decile['y_lasso'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_lgbm_percent = [(decile[decile['y_lgbm'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_true_percent = [(decile[decile['y_test'] < t].shape[0] / total_instances) * 100 for t in thresholds]

# --- Code Block ---
# Creating the Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_accepts_lasso_percent, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=cumulative_accepts_lgbm_percent, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
'''fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_true, mode='lines+markers',
                         name='Cumulative Mean True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))'''

fig.update_layout(
    title='Mean a_to_e Values vs Acceptance Rate',
    xaxis_title='Acceptance Rate',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
#calculate acceptance rate vs mean a_to_e value for Affiliate Conversion vs new models
thresholds = np.linspace(0, 2, 300)
total_instances = ae.shape[0]

cumulative_accepts_lasso_percent = [(ae[ae['y_lasso'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_lgbm_percent = [(ae[ae['y_pred'] < t].shape[0] / total_instances) * 100 for t in thresholds]

cumulative_sums_lasso = [ae[ae['y_lasso'] < t]['y_test'].mean() for t in thresholds]
cumulative_sums_lgbm = [ae[ae['y_pred'] < t]['y_test'].mean() for t in thresholds]

# --- Code Block ---
cumulative_accepts_uw = [(ae[ae['Predicted A_TO_E'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_mean_uw = [ae[ae['Predicted A_TO_E'] < t]['y_test'].mean() for t in thresholds]

fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_accepts_lasso_percent, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean - Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=1)))
fig.add_trace(go.Scatter(x=cumulative_accepts_lgbm_percent, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean - LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=1)))
fig.add_trace(go.Scatter(x=cumulative_accepts_uw, y=cumulative_mean_uw,  mode='lines+markers',
                         name='Cumulative Mean - Affilicate Conversion',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=1)))

fig.update_layout(
    title='Mean a_to_e Values vs Acceptance Rate',
    xaxis_title='Acceptance Rate',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

# --- Code Block ---
df['threshold'] = df['y_pred'].apply(lambda x: 1 if x <= 1 else 0)

# --- Code Block ---
pd.crosstab(df['race_eth'], df['threshold'], normalize='index')

# --- Code Block ---
pd.crosstab(df['gender'], df['threshold'], normalize='index')

# --- Code Block ---
contingency_table = pd.crosstab(df['gender'], df['threshold'])

chi2, pvalue, dof, expected = chi2_contingency(contingency_table, correction=False)
print(f'Observed chi2: {chi2:.5f}')
print(f'p-value: {pvalue:.5f}')

# --- Code Block ---
df_control = df[[
 'plan',
 'gender',
 'tobacco_usage_ind',
 'marketing_channel',
 'uw_catg_des',
 'hhd_ind',
 'race_eth',
 'threshold']]
df_control.info()

# --- Code Block ---
#best hyper-parameters
df_trials = pd.read_csv('gbm_all_feas_06.26.csv')
best_trial = df_trials[df_trials['value'] == df_trials['value'].min()]
best_params = best_trial.filter(regex='params_')  # This regex matches columns that start with 'params_'
best_params.columns = best_params.columns.str.replace('params_', '')
best_params = best_params.iloc[0].to_dict()
for param in ['bagging_freq','max_depth','min_data_in_leaf','num_leaves']:
    best_params[param] = int(best_params[param])
best_params

# --- Code Block ---
thresholds = np.linspace(0, 2, 300)
# Calculating cumulative acceptances for each threshold for both models
cumulative_accepts_lasso = [decile[decile['y_lasso'] < t].shape[0] for t in thresholds]
cumulative_accepts_lgbm = [decile[decile['y_lgbm'] < t].shape[0] for t in thresholds]
cumulative_accepts_true = [decile[decile['y_test'] < t].shape[0] for t in thresholds]

# --- Code Block ---
# Total number of instances
total_instances = decile.shape[0]

# Calculating cumulative acceptances as percentages for each threshold
cumulative_accepts_lasso_percent = [(decile[decile['y_lasso'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_lgbm_percent = [(decile[decile['y_lgbm'] < t].shape[0] / total_instances) * 100 for t in thresholds]
cumulative_accepts_true_percent = [(decile[decile['y_test'] < t].shape[0] / total_instances) * 100 for t in thresholds]

# --- Code Block ---
# Creating the Plotly graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_accepts_lasso_percent, y=cumulative_sums_lasso, mode='lines+markers',
                         name='Cumulative Mean Lasso',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=5)))
fig.add_trace(go.Scatter(x=cumulative_accepts_lgbm_percent, y=cumulative_sums_lgbm, mode='lines+markers',
                         name='Cumulative Mean LGBM',
                         line=dict(color='green', width=2),
                         marker=dict(color='green', size=5)))
'''fig.add_trace(go.Scatter(x=thresholds, y=cumulative_sums_true, mode='lines+markers',
                         name='Cumulative Mean True',
                         line=dict(color='red', width=2),
                         marker=dict(color='red', size=5)))'''

fig.update_layout(
    title='Mean a_to_e Values vs Acceptance Rate',
    xaxis_title='Acceptance Rate',
    yaxis_title='Mean Value',
    legend_title='Models',
    template='plotly_white'
)
fig.show()

