#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Solving problem using ML approach first
# Step 1: Load the dataset
# Step 2: Perform EDA on dataset
# Step 3: Clean the dataset / feature engineering.
# Step 4: Split data into training and test
# Step 5: Train Various ML Models
# Step 6: Hyper parameter tuning using GridSearch
# Step 7: Evaluate the model

# 1. Load the dataset ..We will be using 'iitr-final-project/all_season_summary.csv' for this project.

import pandas as pd

filepath = 'iitr-final-project/all_season_summary.csv'


# In[2]:


# Step 1: Load the dataset
season_summary_data = pd.read_csv(filepath)


# In[3]:


season_summary_data.info()


# In[4]:


# Set the option to display all columns
pd.set_option('display.max_columns', None)
season_summary_data.head()


# In[7]:


season_summary_data.info()


# In[8]:


#Step 2: Perform EDA on dataset

#Checking null values
missing_values = season_summary_data.isnull().sum()


# In[9]:


missing_values


# In[10]:


# Step 3: Clean the dataset
# Feature Engineering
# Extract date features
season_summary_data['start_date'] = pd.to_datetime(season_summary_data['start_date'])
season_summary_data['day'] = season_summary_data['start_date'].dt.day
season_summary_data['month'] = season_summary_data['start_date'].dt.month
season_summary_data['year'] = season_summary_data['start_date'].dt.year


# In[11]:


# Encoding categorical and numerical features.

# Select categorical and numerical features
categorical_features = season_summary_data.select_dtypes(include=['object']).columns.tolist()
numerical_features = season_summary_data.select_dtypes(include=['number']).columns.tolist()

print("Categorical Features:\n", categorical_features)
print("Numerical Features:\n", numerical_features)



# In[12]:


categorical_features_final = ['home_team', 'away_team', 'toss_won', 'decision', 'result','home_score','away_score']
numerical_features_final = ['season','home_overs','away_overs']


# In[13]:


# Define preprocessing steps for categorical and numerical features
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


# In[14]:


# Combine preprocessing steps for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features_final),
        ('num', numerical_transformer, numerical_features_final)
    ])


# In[15]:


# Define features and target variable
X = season_summary_data.drop(columns = ['winner','start_date']) #features
y = season_summary_data['winner'] #target variable


# In[16]:


# Step 4 : Train Test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# In[17]:


season_summary_data.shape


# In[18]:


# Combine preprocessing and modeling steps into a single pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Preprocess training and testing data
X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)


# In[19]:


#y_train.shape
X_train_preprocessed.shape
#X_test_preprocessed.shape
#season_summary_data.shape


# In[20]:


# Checkinmg those 1830 features created after transformations

# Get the unique categories of the categorical features
categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_
unique_cat_features = [f"{feature}_{value}" for feature, values in zip(categorical_features_final, categories) for value in values]

# Get the transformed numerical features
transformed_num_features = numerical_features_final

# Combine the transformed categorical and numerical features
transformed_features = unique_cat_features + transformed_num_features

# Print the transformed features
print(transformed_features)


# In[21]:


#Step 5: Train various models on dataset
# 'C' is hyprparameter for regularizing L2
# 'lbfgs' is Byoden-Fletcher-Goldfarb-Shanno(BFGS) algorithm
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(solver = 'lbfgs', C=10, random_state = 42)
log_clf.fit(X_train_preprocessed,y_train)


# In[22]:


y_train_predict = log_clf.predict(X_train_preprocessed)


# In[23]:


y_train_predict[10]


# In[24]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score

log_accuracy = accuracy_score(y_train_predict,y_train)
log_precision = precision_score(y_train_predict,y_train, average='weighted')
log_recall = recall_score(y_train_predict,y_train, average='weighted')
log_f1 = f1_score(y_train_predict,y_train, average='weighted')


# In[25]:


print(log_accuracy)


# In[26]:


print(log_precision)


# In[27]:


print(log_recall)


# In[28]:


print(log_f1)


# In[29]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators = 20, max_depth =10, random_state=42)
rnd_clf.fit(X_train_preprocessed,y_train)


# In[30]:


y_train_predict = rnd_clf.predict(X_train_preprocessed)


# In[31]:


y_train_predict[100]


# In[32]:


rnd_clf_accuracy = accuracy_score(y_train_predict,y_train)
rnd_clf_precision = precision_score(y_train_predict,y_train, average='weighted')
rnd_clf_recall = recall_score(y_train_predict,y_train, average='weighted')
rnd_clf_f1 = f1_score(y_train_predict,y_train, average='weighted')


# In[33]:


print(rnd_clf_accuracy)


# In[34]:


print(rnd_clf_precision)


# In[35]:


print(rnd_clf_recall)


# In[36]:


print(rnd_clf_f1)


# In[37]:


# Now let's do cross validation in order to verify if our models are not overfitting or underfitting

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

def display_scores(scores):
    print(scores)


# In[38]:


log_clf = LogisticRegression(solver = 'lbfgs', C=10, random_state = 42)


# In[39]:


log_cv_scores = cross_val_score(log_clf,X_train_preprocessed,y_train,cv=3,scoring ='accuracy')


# In[40]:


display_scores(log_cv_scores)


# In[41]:


log_cv_accuracy = log_cv_scores.mean()


# In[42]:


y_train_pred = cross_val_predict(log_clf,X_train_preprocessed,y_train,cv=3)


# In[43]:


confusion_matrix(y_train,y_train_pred)


# In[44]:


log_cv_precision = precision_score(y_train,y_train_pred,average = 'weighted')
log_cv_recall = recall_score(y_train,y_train_pred,average = 'weighted')
log_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')


# In[45]:


log_cv_accuracy


# In[46]:


log_cv_precision


# In[47]:


log_cv_recall


# In[48]:


log_cv_f1_score


# In[49]:


rnd_clf = RandomForestClassifier(n_estimators = 20, max_depth =10, random_state=42)
rnd_cv_scores = cross_val_score(rnd_clf,X_train_preprocessed,y_train,cv=3,scoring ='accuracy')


# In[50]:


display_scores(rnd_cv_scores)


# In[51]:


rnd_cv_accuracy = rnd_cv_scores.mean()


# In[52]:


y_train_pred = cross_val_predict(rnd_clf,X_train_preprocessed,y_train,cv=3)


# In[53]:


confusion_matrix(y_train,y_train_pred)


# In[54]:


rnd_cv_precision = precision_score(y_train,y_train_pred,average = 'weighted')
rnd_cv_recall = recall_score(y_train,y_train_pred,average = 'weighted')
rnd_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')


# In[55]:


rnd_cv_accuracy


# In[56]:


rnd_cv_precision


# In[57]:


rnd_cv_recall


# In[58]:


rnd_cv_f1_score


# In[59]:


print("=== Softmax/Logistic Regression === ")
display_scores(log_cv_scores)
print("log_cv_accuracy:", log_cv_accuracy)
print("log_cv_precision:", log_cv_precision)
print("log_cv_recall:", log_cv_recall)
print("log_cv_f1_score:", log_cv_f1_score)

print("=== Random Forest === ")
display_scores(rnd_cv_scores)
print("rnd_cv_accuracy:", rnd_cv_accuracy)
print("rnd_cv_precision:", rnd_cv_precision)
print("rnd_cv_recall :", rnd_cv_recall )
print("rnd_cv_f1_score:", rnd_cv_f1_score)


# In[60]:


#Step 6: Hyperparameter tuning using Gridsearch
from sklearn.ensemble import VotingClassifier   # model made up of two models i.e random forest and logistic regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "lr__solver":["lbfgs"],
        "lr__C":[5],
        "rf__n_estimators":[20],
        "rf__max_depth":[10,15],
    }]


# In[61]:


log_clf_ens = LogisticRegression(solver="lbfgs", C=10 , random_state=42 )


# In[62]:


rnd_clf_ens = RandomForestClassifier(n_estimators=20, max_depth=10 , random_state=42)


# In[63]:


voting_clf_grid_search = VotingClassifier(
    estimators=[('lr', log_clf_ens), ('rf', rnd_clf_ens)],
    voting='soft')


# In[64]:


grid_search = GridSearchCV(voting_clf_grid_search, param_grid, cv=3, scoring='neg_mean_squared_error')


# In[65]:


#Value error before encoding: The error indicates that there are string values in your training labels (y_train) that cannot be converted to float. 
#This typically occurs when your target variable contains non-numeric values, which is common in classification tasks.
#To address this issue, ensure that your target variable (y_train) contains numerical labels representing the classes rather than string labels. If your target variable contains string labels (e.g., 'SRH', 'MI', etc.), you need to encode them into numerical values before fitting the model.
#Here's how you can encode categorical labels using LabelEncoder:

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable
y_train_encoded = label_encoder.fit_transform(y_train)

# Now, you can use y_train_encoded for fitting your model
grid_search.fit(X_train_preprocessed, y_train_encoded)


# In[66]:


grid_search.best_params_


# In[67]:


grid_search.best_estimator_


# In[68]:


# Current doubt : why input shape i.e columns changed to 1830 after preprocessing/pipeline transformation.?
# clarified now happens because of transformations applied.


#The increase in the number of features from 48 to 1830 after preprocessing can be attributed to the transformation applied to your categorical features.

#Here's a breakdown of what might have happened:

#One-Hot Encoding: If your categorical features had a large number of unique categories, or if you used one-hot encoding to represent categorical variables, this could significantly increase the number of features. Each unique category in a categorical feature gets transformed into a binary column, resulting in additional columns in your dataset.

#Expansion of Categorical Features: The preprocessing steps might have expanded the categorical features in some way, such as creating new features through feature engineering or interactions between categorical variables and numerical variables. For example, if you had interaction terms between categorical variables or polynomial features, this could lead to an increase in the number of features.

#Missing Values Handling: If there were missing values in your categorical features, the preprocessing steps might have imputed or filled in these missing values using techniques like mean or mode imputation, which could result in additional features.

#To understand exactly why the number of features increased, you can inspect the output of your preprocessing steps, including the transformed categorical features. This will help you identify which transformations were applied and why they led to the increase in feature dimensionality. Additionally, you can review your preprocessing code to check for any unintended transformations or encoding schemes that might have contributed to the expansion of features.


# In[69]:


#Steps for Neural Network model

#1. Preprocesss the data : Handle missing values and encode categorical variables.
#2. Split data into training and testing sets.
#3. Build neural network model
#4. Compile the model
#5. Train the model
#6. Evaluate the model


# In[70]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

# Load the dataset
data = pd.read_csv("iitr-final-project/all_season_summary.csv")  # Replace "ipl_dataset.csv" with your dataset filename

# Drop columns not relevant for prediction
data = data.drop(columns=['id', 'name', 'short_name', 'description', 'start_date', 'end_date', 'venue_id', 'venue_name', 
                          'highlights', 'match_days', 'umpire1', 'umpire2', 'tv_umpire', 'referee', 'reserve_umpire','1st_inning_score',
                         '2nd_inning_score','home_score','away_score'])

# Handle missing values
data = data.dropna()

# Encode categorical variables
cat_cols = ['home_team', 'away_team', 'toss_won', 'decision', 'winner', 'result', 'home_captain', 'away_captain',
            'pom', 'points', 'super_over', 'home_playx1', 'away_playx1', 'home_key_batsman', 'home_key_bowler',
            'away_key_batsman', 'away_key_bowler']

encoder = LabelEncoder()
for col in cat_cols:
    data[col] = encoder.fit_transform(data[col])

# Convert object dtype columns to numeric
data = data.apply(pd.to_numeric, errors='ignore')

# Split the data into features and target
X = data.drop(columns=['winner'])
y = data['winner']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
#model = Sequential([
#    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#    Dense(64, activation='relu'),
#    Dense(len(y_train.unique()), activation='softmax')  # Number of unique winners as output neurons
#])

print(X_train_scaled.shape)  # (806,25)
print(X_train_scaled.shape[1]) # we will send 25 as input because these are features and will act as neuron
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape =(X_train_scaled.shape[1],)),
    keras.layers.Dense(300,kernel_initializer = 'he_normal',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(300,kernel_initializer = 'he_normal', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(300,kernel_initializer = 'he_normal', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(100,kernel_initializer = 'he_normal', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(100,kernel_initializer = 'he_normal', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(len(y_train.unique()),activation = 'softmax')  # multi classs classification hence using softmax,if binary simoid is used
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',     # multi class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=15, validation_split=0.2)

# Evaluate the model
#test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
#print(f'Test accuracy: {test_acc}')


# In[71]:


model.summary()


# In[ ]:





# In[ ]:




