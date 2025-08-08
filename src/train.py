# --- Code Block ---
# Sliding window validation
'''for file in file_names:
        # Load dataframes for the current training window
        train_df = pd.read_feather(directory_path + file)
        train_df = train_df.head(1000)
        train_df.to_feather(directory_path + 'test' + file)'''

# --- Code Block ---
results_df = study.trials_dataframe()

# Calculate and add average metrics to the dataframe
for metric in ['mse', 'rmse', 'r2']:
    train_metrics = results_df.filter(like=f'train_{metric}').mean(axis=1)
    test_metrics = results_df.filter(like=f'test_{metric}').mean(axis=1)
    results_df[f'average_train_{metric}'] = train_metrics
    results_df[f'average_test_{metric}'] = test_metrics

# --- Code Block ---
fig = px.line(results_df, x="number", y=["average_train_r2","average_test_r2"], 
              title='Train R2 vs Test R2', 
              labels={"value": "R2", "variable": "R2 Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")
fig.show()

# --- Code Block ---
results_df = study.trials_dataframe()

# Calculate and add average metrics to the dataframe
for metric in ['mse', 'rmse', 'r2']:
    train_metrics = results_df.filter(like=f'train_{metric}').mean(axis=1)
    test_metrics = results_df.filter(like=f'test_{metric}').mean(axis=1)
    results_df[f'average_train_{metric}'] = train_metrics
    results_df[f'average_test_{metric}'] = test_metrics

results_df

# --- Code Block ---
results_df = study2.trials_dataframe()

# Calculate and add average metrics to the dataframe
for metric in ['mse', 'rmse', 'r2']:
    train_metrics = results_df.filter(like=f'train_{metric}').mean(axis=1)
    test_metrics = results_df.filter(like=f'test_{metric}').mean(axis=1)
    results_df[f'average_train_{metric}'] = train_metrics
    results_df[f'average_test_{metric}'] = test_metrics

# --- Code Block ---
fig = px.line(results_df, x="number", y=["average_train_rmse","average_test_rmse"], 
              title='Train RMSE vs Test RMSE - eTRAM', 
              labels={"value": "RMSE", "variable": "RMSE Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")
fig.show()

# --- Code Block ---
fig = px.line(results_df, x="number", y=["average_train_r2","average_test_r2"], 
              title='Train R2 vs Test R2 - eTram', 
              labels={"value": "R2", "variable": "R2 Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")
fig.show()

# --- Code Block ---
results_df = study3.trials_dataframe()

# Calculate and add average metrics to the dataframe
for metric in ['mse', 'rmse', 'r2']:
    train_metrics = results_df.filter(like=f'train_{metric}').mean(axis=1)
    test_metrics = results_df.filter(like=f'test_{metric}').mean(axis=1)
    results_df[f'average_train_{metric}'] = train_metrics
    results_df[f'average_test_{metric}'] = test_metrics

# --- Code Block ---
fig = px.line(results_df, x="number", y=["average_train_rmse","average_test_rmse"], 
              title='LightGBM: Train RMSE vs Test RMSE - eTram', 
              labels={"value": "RMSE", "variable": "RMSE Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")

# --- Code Block ---
fig = px.line(results_df, x="number", y=["average_train_r2","average_test_r2"], 
              title='LightGBM: Train R2 vs Test R2 - eTram', 
              labels={"value": "R2", "variable": "R2 Type"}, 
              markers=True)
fig.update_traces(mode="markers+lines")

