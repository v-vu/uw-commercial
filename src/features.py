# --- Code Block ---
all_correlations.reset_index(inplace=True)
all_correlations.columns = ['features', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5']

# --- Code Block ---
# Reshape the DataFrame to long format
df_long = all_correlations.melt(id_vars='features', var_name='fold', value_name='value')
df_long = df_long[df_long['features'] != 'a_to_e_value']

# Calculate the absolute values
df_long['abs_value'] = df_long['value'].abs()

# Sort by absolute value to find top values across all folds and features
df_long_sorted = df_long.sort_values(by='abs_value', ascending=False)

# Plot the top absolute values using Plotly
fig = px.bar(df_long_sorted.head(100), x='features', y='abs_value', color='fold', 
             title='Top Absolute Values for Each Feature and Fold', 
             labels={'abs_value': 'Absolute Value', 'features': 'Features'},
             text='value')
fig.show()

# --- Code Block ---
fig = px.bar(df_long_sorted.iloc[100:200], x='features', y='abs_value', color='fold', 
             title='Top Absolute Values for Each Feature and Fold', 
             labels={'abs_value': 'Absolute Value', 'features': 'Features'},
             text='value')
fig.show()

# --- Code Block ---
fig = px.bar(df_long_sorted.iloc[200:300], x='features', y='abs_value', color='fold', 
             title='Top Absolute Values for Each Feature and Fold', 
             labels={'abs_value': 'Absolute Value', 'features': 'Features'},
             text='value')
fig.show()

# --- Code Block ---
# Read the contents of the file back into a list
with open('drop_features.txt', 'r') as file:
    drop_features = [line.strip() for line in file]

# --- Code Block ---
mean_corr = pd.DataFrame(df_long.groupby(by=['features'])['abs_value'].mean().sort_values(ascending = False)).reset_index()

# --- Code Block ---
mean_corr[mean_corr['features'].str.startswith('ccs_')].head(10)
#240 - Medications (Injections, infusions and other forms)
#224 - Cancer chemotherapy
#227 - Consultation, evaluation, and preventive care
#233 - Laboratory - Chemistry and Hematology
#231 - Other therapeutic procedures
#243 - DME and supplies
#91 - Peritoneal dialsysis
#178 - CT scan chest
#183 - Routine chest X-ray

# --- Code Block ---
mean_corr[mean_corr['features'].str.startswith('ccsr_')].head(15)
#gen003 - Chronic kidney disease
#bld003 - Aplastic anemia
#gen006 - Diseases of kidney and ureters
#neo070 - Secondary malignancies
#bld001 - Nutritional anemia
#cir007 - Essential hypertension
#neo022 - Respiratory cancers
#end011 - Fluid and electrolyte disorders
#end003 - Diabetes mellitus with complication
#end016 - Nutritional and metabolic disorders
#end010 - Disorders of lipid metabolism
#cir008 - Hypertension with complications and secondary hypertension
#nvs018 - Myopathies

# --- Code Block ---
mean_corr[mean_corr['features'].str.startswith('drg_')].head(10)
#871 - Septicemia or severe sepsis without MV >96 hours with MCC
#314 - Other circulatory system diagnoses with MCC
#291 - Heart failure and shock with MCC
#166 - Other respiratory system O.R. procedures with MCC
#189 - Pulmonary edema and respiratory failure
#682 - Renal failure with MCC
#813 - Coagulation disorders
#193 - Simple pneumonia and pleurisy with MCC
#673 - Other kidney and urinary tract procedures with MCC
#003 - ECMO or tracheostomy with MV >96 hours or PDX except face, mouth and neck with major O.R. procedure

# --- Code Block ---
#Top 15 most correlated CCS to top 10 CCSR

for feature in top_10_ccsr_3mos:
    ccsr_corr = df[df.select_dtypes(include=['number']).columns].corrwith(df[feature])
    ccsr_corr= pd.DataFrame(ccsr_corr, columns=['corr'])
    ccsr_corr['abs_value'] = ccsr_corr['corr'].abs()
    ccsr_corr = ccsr_corr[ccsr_corr.index.str.startswith('ccs_')].sort_values(by = 'abs_value', ascending=False).head(15).reset_index()
    ccsr_corr.rename(columns={'index': 'feature'}, inplace=True)

    # Create a bar chart using Plotly Express
    fig = px.bar(ccsr_corr, y='feature', x='abs_value', title='Correlation by Feature: ' + feature, orientation='h')

    # Improve layout for better readability
    #fig.update_layout(xaxis_title='Features', yaxis_title='Correlation', xaxis={'categoryorder':'total descending'})
    fig.update_layout(yaxis={"dtick":1},margin={"t":100,"b":100},height=800)
    
    fig.show()

# --- Code Block ---
fig = px.line(results_df, x="number", y=["average_train_rmse","average_test_rmse"], 
              title='LightGBM: Train RMSE vs Test RMSE - all features', 
              labels={"value": "RMSE", "variable": "RMSE Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")

# --- Code Block ---
fig = px.line(results_df, x="number", y=["average_train_r2","average_test_r2"], 
              title='LightGBM: Train R2 vs Test R2 - all features', 
              labels={"value": "R2", "variable": "R2 Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")

# --- Code Block ---
features = X_test.columns.tolist()
# Open a file in write mode
with open('features.txt', 'w') as file:
    # Write each item on a new line
    for item in features:
        file.write(f"{item}\n")

# --- Code Block ---
with open('features.txt', 'r') as file:
    feas = [line.strip() for line in file.readlines()]

