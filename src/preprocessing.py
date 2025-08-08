# --- Code Block ---
df = pd.read_feather(path + '\Data\\train 2018-01-01.feather')

#downcast float and int column
for col in df.select_dtypes(include=['float64', 'int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
#remove trailing spaces
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()
        
df = df[df['plan'].isin(['F', 'G', 'N'])]
df.drop(columns=['policy_agreement_id'],inplace=True)

numeric_cols = df.select_dtypes(include=['number']).columns
#ccsr_columns = [col for col in df.columns if 'ccsr_' in col]
#ccs_columns = [col for col in df.columns if 'ccs_' in col]
#drg_columns = [col for col in df.columns if 'drg_' in col]

missing_percentage = df[numeric_cols].isnull().mean() * 100
na_cols = missing_percentage[missing_percentage==100].index.tolist()

# --- Code Block ---
#check for columns with 100% missing values in all the datasets
files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('.feather')]

missing_status = {} 
# Loop through each feather file, read it, and append to the list
for file in files:
    filepath = os.path.join(path + '\Data\\', file)
    z = pd.read_feather(filepath,columns=na_cols)

    for col in z.columns:
        # If first dataframe, initialize the column status
        if file == files[0]:
            missing_status[col] = df[col].isnull().all()
        else:
            # Update status only if previously identified as all missing
            if missing_status[col]:
                missing_status[col] = df[col].isnull().all()

# After processing all dataframes, filter columns that are 100% missing across all
all_missing_columns = [col for col, is_all_missing in missing_status.items() if is_all_missing]

print("Columns with 100% missing values in all dataframes:", all_missing_columns)

# --- Code Block ---
#correlations
files = [f for f in os.listdir(path + '\Data\\' ) if f.endswith('.feather')]

all_correlations = []
for file in files:
    filepath = os.path.join(path + '\Data\\', file)
    df = pd.read_feather(filepath)
    df = df.drop(columns=all_missing_columns)
    df.fillna(0, inplace=True)

    correlation_matrix = df[df.select_dtypes(include=['number']).columns].corrwith(df['a_to_e_value'])
    correlation_matrix.sort_values(ascending=False)
    
    
    all_correlations.append(correlation_matrix)

# --- Code Block ---
#identify features that have null correlations
null_feas = df_long_sorted[df_long_sorted.isnull().any(axis=1)][['features','fold']].groupby('features')['fold'].nunique()
null_feas = null_feas[null_feas == 5].index.tolist()
drop_features = all_missing_columns + null_feas

# Export combined list to a file
with open('drop_features.txt', 'w') as file:
    file.writelines(f"{item}\n" for item in drop_features)

# --- Code Block ---
df = pd.read_feather(path + '\Data\\train 2022-01-01.feather')
df.fillna(0, inplace=True)

# Read the contents of the file back into a list
with open('drop_features.txt', 'r') as file:
    drop_features = [line.strip() for line in file]
df = df.drop(columns=drop_features, errors = 'ignore')

# --- Code Block ---
df = pd.read_feather(r"\\file018\PMO_GRPB\RENEWAL\CMS Data\SHDS\AC Replacement Model\Data\fold5.feather")

# --- Code Block ---
#downcast float and int column and to sparse
for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
#Clean object columns
for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].astype(str)

df.fillna(0, inplace=True)

# --- Code Block ---
pol.fillna(0, inplace=True)

# --- Code Block ---
#downcast float and int column and to sparse
for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
#Clean object columns
for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].astype(str)

df.fillna(0, inplace=True)

