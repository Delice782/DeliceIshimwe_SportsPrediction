#!/usr/bin/env python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

#### Data Preprocessing 

"""
This function loads CSV files, cleans the data: removing duplicates and filling missing values. 
It takes dataset_address and encoding as parameters and returns a cleaned dataframe.
"""

import pandas as pd

def load_clean_dataset(dataset_address, encoding='utf-8'):
    
    df = pd.read_csv(dataset_address, encoding=encoding, on_bad_lines='skip')
    
    df.drop_duplicates(inplace=True)
    
    for data in df.columns:
        if df[data].dtype == 'object':
            df[data].fillna(df[data].mode()[0], inplace=True)
        else:
            df[data].fillna(df[data].median(), inplace=True)
    
    return df

training_dataset = "C:\\Users\\user\\Downloads\\male_players (legacy).csv"
testing_dataset = "C:\\Users\\user\\Downloads\\players_22 (1).csv"

cleaned_train_df = load_clean_dataset(training_dataset)
cleaned_test_df = load_clean_dataset(testing_dataset)

# Display the first 5 rows of the cleaned DataFrame
cleaned_train_df.head()

# Display the first 5 rows of the cleaned DataFrame
cleaned_test_df.head()

# Display the shape of the cleaned DataFrame
cleaned_train_df.shape

# Display the shape of the cleaned DataFrame
cleaned_test_df.shape

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This method performs Exploratory Data Analysis on df dataFrame.
It takes df dataFrame as a parameter which contains data to be analyzed.
It focus on numeric columns through displaying dataset information and sumamry statistics, 
providing correlation heatmap for visualizing the relationships between pair of variables, 
and  creates scatterplot matrix for visualizing relationships between pairs across multiple variables.
It also generates the distribution of the target variable 'overall' rating column and
visualize the count of players by their preferred foot, if they are present. 
This illustrates how ratings are distributed among players.
"""

def conduct_EDA(df):
    num_df = df.select_dtypes(include='number')
    
    print("Dataset info:")
    print(num_df.info())
    
    print("\nSummary statistics:")
    print(num_df.describe())
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()
    
    sns.pairplot(num_df[['overall', 'age', 'height_cm', 'weight_kg', 'value_eur', 'wage_eur']])
    plt.suptitle('Pairplot of Numerical Variables', y=1.02)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(df['overall'], bins=20, kde=True)
    plt.title('Distribution of Overall Ratings')
    plt.xlabel('Overall Rating')
    plt.ylabel('Count')
    plt.show()
    
    if 'preferred_foot' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='preferred_foot', data=df)
        plt.title('Count of Players by Preferred Foot')
        plt.xlabel('Preferred Foot')
        plt.ylabel('Count')
        plt.show()
        
# Conduct EDA for cleaned train dataset
conduct_EDA(cleaned_train_df)

# Conduct EDA for cleaned test dataset
conduct_EDA(cleaned_test_df)

"""
This Function takes df dataframe as parameter and 
it removes useless variables and returns the DataFrame with the remaining columns.
"""

def feature_selection(df):
    print("Original columns:")
    print(df.columns)

    # List of useless variables.
    useless_variables = ['fifa_version', 'fifa_update', 'fifa_update_date', 'league_id', 'club_contract_valid_until_year']
    
    # Check which columns in useless_variables are not present in df.columns
    dropped_columns = [column for column in useless_variables if column in df.columns]
    
    if dropped_columns:
        df.drop(columns=dropped_columns, inplace=True)
        print(f"Dropped columns: {dropped_columns}")
    else:
        print("No useless variables are found in DataFrame.")
    
    # selecting all the remaining columns
    selected_features = df.columns.tolist() 
    
    return df[selected_features]


# In[13]:


# Select features for the training dataset
selected_training_df = feature_selection(cleaned_train_df) 

# Select features for the testing dataset
selected_testing_df = feature_selection(cleaned_test_df)


#### Feature Engineering

from sklearn.preprocessing import StandardScaler

"""
This function takes df dataFrame, target variable 'overall', and the threshols=0.2 as parameters.
It selects features that have a correlation coefficient above the threshold with the target variable.
It returns List of selected features based on correlation threshold.
"""

def choose_correlated_features(df, target='overall', threshold=0.2):
    num_df = df.select_dtypes(include=['number'])
    corr_matrix = num_df.corr()
    corr_with_target_variable = corr_matrix[target].abs().sort_values(ascending=False)
    selected_features = corr_with_target_variable[corr_with_target_variable >= threshold].index.tolist()
    
    return selected_features

from sklearn.preprocessing import StandardScaler

"""
This function ensures that numerical features in 
both training and testing datasets are scaled similarly using StandardScaler.
It takes Training DataFrame containing selected features, testing DataFrame containing selected features,
and List of feature names to scale as parameters and 
returns Scaled training DataFrame and Scaled testing DataFrame.
"""

def scale_features(train_df, test_df, features):
    
    scaler = StandardScaler()

    scaled_training_df = train_df.copy()
    scaled_testing_df = test_df.copy()

    scaled_testing_df = scaled_testing_df.reindex(columns=scaled_training_df.columns, fill_value=0)
    
    scaled_training_df[features] = scaler.fit_transform(selected_training_df[features])
    scaled_testing_df[features] = scaler.transform(scaled_testing_df[features])
    
    return scaled_training_df, scaled_testing_df

# In[16]:

# Picking features correlated with 'overall' rating
selected_features = choose_correlated_features(selected_training_df, target='overall', threshold=0.2)

# Scale selected features
scaled_training_df, scaled_testing_df = scale_features(selected_training_df, selected_testing_df, selected_features)


#### Training Models

# In[17]:


# Training and Testing scaled datasets
X_train = scaled_training_df.drop('overall', axis=1) 
y_train = scaled_training_df['overall']  

X_test = scaled_testing_df.drop('overall', axis=1) 
y_test = scaled_testing_df['overall'] 


# In[18]:


X_train.columns.tolist()


# In[19]:


"""
This function extracts and returns column names from the list 'categorical_variables'
that are present in the 'df' DataFrame.
"""

categorical_variables = [
    'player_url', 'fifa_update', 'fifa_update_date', 'short_name', 'long_name', 
    'player_positions', 'dob', 'league_name', 'league_level', 'club_name', 
    'club_position', 'club_loaned_from', 'club_joined_date', 'club_contract_valid_until_year', 
    'nationality_id', 'nationality_name', 'nation_position', 'preferred_foot', 'weak_foot', 
    'skill_moves', 'international_reputation', 'work_rate', 'body_type', 'real_face', 
    'release_clause_eur', 'player_tags', 'player_traits', 'mentality_composure', 'player_face_url'
]

def find_categ_columns(df):
    new_categ_columns = []
    for column in categorical_variables:
        if column in df.columns:
            new_categ_columns.append(column)
            
    return new_categ_columns
            


# In[20]:

new_columns_train = find_categ_columns(X_train)
new_columns_train


# In[21]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
This method reduces the number of unique categorical values for preventing overfitting in models.
It gets the frequency distribution of the column and 
when the number of unique categorical values is greater than the threshold, 
it replaces less frequent values with 'Other'
"""
def reduce_unique_categ_values(df, categorical_variables, threshold=100):
    for column in categorical_variables:
        if column in df.columns:
            value_counts = df[column].value_counts()
            if len(value_counts) > threshold:
                categ_values_to_keep = value_counts.index[:threshold]
                df[column] = df[column].where(df[column].isin(categ_values_to_keep), 'Other')
    return df

# In[22]:


"""
This method identifies and returns a list of column names from DataFrame 'df'
that are categorical basing on their data type or when they are categorical and
have a high number of unique categories relative to the total number of observations 'threshold'.
"""

def identify_categorical_columns(df, threshold=0.9):
    categorical_columns = []
    for column in df.columns:
        if df[column].dtype == 'object' or (df[column].dtype == 'category' and len(df[column].cat.categories) > threshold * len(df)):
            categorical_columns.append(column)
    return categorical_columns

# In[23]:


"""
This method performs one-hot-encoding on categorical columns in the a DataFrame 'df'
It takes DataFrame 'df' to be encoded and maximum number of unique categories '100' as parameters, 
and returns DataFrame 'df_encoded' after encoding of categorical columns.
"""

def one_hot_encode_dataframe(df, max_categories=100):

    df = df.apply(lambda col: col.astype(str) if col.dtype == 'object' or col.dtype == 'float' else col)
    
    categorical_columns = identify_categorical_columns(df)
    
    df = reduce_unique_categ_values(df, categorical_columns, threshold=max_categories)
    
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    
    encoded_sparse_matrix = encoder.fit_transform(df[categorical_columns])
    
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_sparse_matrix, columns=encoder.get_feature_names_out(categorical_columns))
    
    numeric_df = df.select_dtypes(exclude=['object', 'category'])
    
    df_encoded = pd.concat([numeric_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    return df_encoded


# In[24]:


# Applying one-hot encoding
X_train_encoded = one_hot_encode_dataframe(X_train)
X_test_encoded = one_hot_encode_dataframe(X_test)


# In[25]:

X_train_encoded


# In[26]:


X_test_encoded


# In[27]:


# Display the shape of the encoded training dataFrame
print(X_train_encoded.shape)
print(y_train.shape)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Initialize the regressor models
rand_forest_regressor = RandomForestRegressor(random_state=42)
xgb_regressor = XGBRegressor(random_state=42)
grad_boost_regressor = GradientBoostingRegressor(random_state=42)

# List of models for iteration
models = [('Random Forest', rand_forest_regressor),
          ('XGBoost', xgb_regressor),
          ('Gradient Boosting', grad_boost_regressor)]

# Conduct cross-validation and evaluate models
for name, model in models:
    scores = cross_val_score(model, X_train_encoded, y_train, cv=5, scoring='r2')
    print(f"{name} R^2 Score: {scores.mean():.4f} (±{scores.std():.4f})")

    # fit the model on the entire training set
    model.fit(X_train_encoded, y_train)


#### Testing and Evaluation

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Instantiating the regressor models
rand_forest_regressor = RandomForestRegressor(random_state=42)
xgb_regressor = XGBRegressor(random_state=42)
grad_boost_regressor = GradientBoostingRegressor(random_state=42)

# Training the models
rand_forest_regressor.fit(X_train, y_train)
xgb_regressor.fit(X_train, y_train)
grad_boost_regressor.fit(X_train, y_train)

# Predict using trained models
rand_forest_predictions = rand_forest_regressor.predict(X_test)
xgb_predictions = xgb_regressor.predict(X_test)
grad_boost_predictions = grad_boost_regressor.predict(X_test)

# Evaluate the  performance
rand_forest_r2 = r2_score(y_test, rand_forest_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)
grad_boost_r2 = r2_score(y_test, grad_boost_predictions)

rand_forest_mse = mean_squared_error(y_test, rand_forest_predictions)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
grad_boost_mse = mean_squared_error(y_test, grad_boost_predictions)

# Dislay performance metrics
print(f"Random Forest: R^2 = {rand_forest_r2:.4f}, MSE = {rand_forest_mse:.4f}")
print(f"XGBoost: R^2 = {xgb_r2:.4f}, MSE = {xgb_mse:.4f}")
print(f"Gradient Boosting: R^2 = {grad_boost_r2:.4f}, MSE = {grad_boost_mse:.4f}")
