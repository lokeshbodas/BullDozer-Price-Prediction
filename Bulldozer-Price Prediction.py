#!/usr/bin/env python
# coding: utf-8

# # Predicting the Sale Price of Bulldozers using Machine Learning
# 
# 
# ## Problem definition
# 
# > How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[2]:


# Import training and validation sets
df = pd.read_csv("TrainAndValid.csv",
                 low_memory=False)


# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


df.columns


# In[6]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])


# In[7]:


df.saledate[:1000]


# In[8]:


df.saledate.dtype


# In[9]:


df.SalePrice.plot.hist()


# ### Parsing dates
# 
# When we work with time series data, we want to enrich the time & date component as much as possible.
# 
# We can do that by telling pandas which of our columns has dates in it using the `parse_dates` parameter.

# In[10]:


# Import data again but this time parse dates
df = pd.read_csv("TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=["saledate"])


# In[11]:


df.saledate.dtype


# In[12]:


df.saledate[:1000]


# In[13]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])


# In[14]:


df.head()


# In[15]:


df.head().T


# In[16]:


df.saledate.head(20)


# ### Sort DataFrame by saledate
# 
# When working with time series data, it's a good idea to sort it by date.

# In[17]:


# Sort DataFrame in date order
df.sort_values(by=["saledate"], inplace=True, ascending=True)
df.saledate.head(20)


# ### Make a copy of the original DataFrame
# 
# We make a copy of the original dataframe so when we manipulate the copy, we've still got our original data.

# In[18]:


# Make a copy of the original DataFrame to perform edits on
df_tmp = df.copy()


# ### Add datetime parameters for `saledate` column

# In[19]:


df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear


# In[20]:


df_tmp.head().T


# In[21]:


# Now we've enriched our DataFrame with date time features, we can remove 'saledate'
df_tmp.drop("saledate", axis=1, inplace=True)


# In[22]:


# Check the values of different columns
df_tmp.state.value_counts()


# In[23]:


df_tmp.head()


# In[24]:


len(df_tmp)


# ## 5. Modelling 

# In[26]:


df_tmp.info()


# In[27]:


df_tmp["UsageBand"].dtype


# In[28]:


df_tmp.isna().sum()


# ### Convert string to categories
# 
# One way we can turn all of our data into numbers is by converting them into pandas catgories.
# 
# We can check the different datatypes compatible with pandas here: https://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html#data-types-related-functionality

# In[29]:


df_tmp.head().T


# In[30]:


pd.api.types.is_string_dtype(df_tmp["UsageBand"])


# In[31]:


# Find the columns which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[32]:


# If you're wondering what df.items() does, here's an example
random_dict = {"key1": "hello",
               "key2": "world!"}

for key, value in random_dict.items():
    print(f"this is a key: {key}",
          f"this is a value: {value}")


# In[33]:


# This will turn all of the string value into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()


# In[34]:


df_tmp.info()


# In[35]:


df_tmp.state.cat.categories


# In[36]:


df_tmp.state.cat.codes


# In[37]:


# Check missing data
df_tmp.isnull().sum()/len(df_tmp)


# ### Save preprocessed data

# In[38]:


# Export current tmp dataframe
df_tmp.to_csv("train_tmp.csv",
              index=False)


# In[39]:


# Import preprocessed data
df_tmp = pd.read_csv("train_tmp.csv",
                     low_memory=False)
df_tmp.head().T


# In[40]:


df_tmp.isna().sum()


# ## Fill missing values 
# 
# ### Fill numerical missing values first

# In[41]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[42]:


df_tmp.ModelID


# In[43]:


# Check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[44]:


# Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())


# In[45]:


# Demonstrate how median is more robust than mean
hundreds = np.full((1000,), 100)
hundreds_billion = np.append(hundreds, 1000000000)
np.mean(hundreds), np.mean(hundreds_billion), np.median(hundreds), np.median(hundreds_billion)


# In[46]:


# Check if there's any null numeric values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[47]:


# Check to see how many examples were missing
df_tmp.auctioneerID_is_missing.value_counts()


# In[48]:


df_tmp.isna().sum()


# ### Filling and turning categorical variables into numbers

# In[49]:


# Check for columns which aren't numeric
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[50]:


# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1
        df_tmp[label] = pd.Categorical(content).codes+1


# In[51]:


pd.Categorical(df_tmp["state"]).codes+1


# In[52]:


df_tmp.info()


# In[53]:


df_tmp.head().T


# In[54]:


df_tmp.isna().sum()


# Now that all of data is numeric as well as our dataframe has no missing values, we should be able to build a machine learning model.

# In[55]:


df_tmp.head()


# In[56]:


len(df_tmp)


# In[57]:


get_ipython().run_cell_magic('time', '', '# Instantiate model\nmodel = RandomForestRegressor(n_jobs=-1,\n                              random_state=42)\n\n# Fit the model\nmodel.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])\n')


# In[59]:


# Score the model
model.score(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])


# ### Splitting data into train/validation sets

# In[60]:


df_tmp.saleYear


# In[61]:


df_tmp.saleYear.value_counts()


# In[62]:


# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)


# In[63]:


# Split data into X & y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[64]:


y_train


# ### Building an evaluation function

# In[65]:


# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    """
    Caculates root mean squared log error between predictions and
    true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_valid, val_preds)}
    return scores


# ## Testing our model on a subset (to tune the hyperparameters)

# In[66]:


len(X_train)


# In[67]:


# Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)


# In[68]:


get_ipython().run_cell_magic('time', '', '# Cutting down on the max number of samples each estimator can see improves training time\nmodel.fit(X_train, y_train)\n')


# In[69]:


(X_train.shape[0] * 100) / 1000000


# In[70]:


10000 * 100


# In[71]:


show_scores(model)


# ### Hyerparameter tuning with RandomizedSearchCV

# In[72]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n\n# Different RandomForestRegressor hyperparameters\nrf_grid = {"n_estimators": np.arange(10, 100, 10),\n           "max_depth": [None, 3, 5, 10],\n           "min_samples_split": np.arange(2, 20, 2),\n           "min_samples_leaf": np.arange(1, 20, 2),\n           "max_features": [0.5, 1, "sqrt", "auto"],\n           "max_samples": [10000]}\n\n# Instantiate RandomizedSearchCV model\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                    random_state=42),\n                              param_distributions=rf_grid,\n                              n_iter=2,\n                              cv=5,\n                              verbose=True)\n\n# Fit the RandomizedSearchCV model\nrs_model.fit(X_train, y_train)\n')


# In[73]:


# Find the best model hyperparameters
rs_model.best_params_


# In[74]:


# Evaluate the RandomizedSearch model
show_scores(rs_model)


# ### Training a model with the best hyperparamters
# 
# **Note:** These were found after 100 iterations of `RandomizedSearchCV`.

# In[75]:


get_ipython().run_cell_magic('time', '', '\n# Most ideal hyperparamters\nideal_model = RandomForestRegressor(n_estimators=40,\n                                    min_samples_leaf=1,\n                                    min_samples_split=14,\n                                    max_features=0.5,\n                                    n_jobs=-1,\n                                    max_samples=None,\n                                    random_state=42) # random state so our results are reproducible\n\n# Fit the ideal model\nideal_model.fit(X_train, y_train)\n')


# In[76]:


# Scores for ideal_model (trained on all the data)
show_scores(ideal_model)


# In[77]:


# Scores on rs_model (only trained on ~10,000 examples)
show_scores(rs_model)


# ## Make predictions on test data

# In[78]:


# Import the test data
df_test = pd.read_csv("Test.csv",
                      low_memory=False,
                      parse_dates=["saledate"])

df_test.head()


# In[79]:


# Make predictions on the test dataset
test_preds = ideal_model.predict(df_test)


# The test data isn't in the same format of our other data, so we have to fix it. Let's create a function to preprocess our data.

# ### Preprocessing the data (getting the test dataset in the same format as our training dataset)

# In[80]:


def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    
    df.drop("saledate", axis=1, inplace=True)
    
    # Fill the numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
    
        # Filled categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # We add +1 to the category code because pandas encodes missing categories as -1
            df[label] = pd.Categorical(content).codes+1
    
    return df


# In[81]:


# Process the test data 
df_test = preprocess_data(df_test)
df_test.head()


# In[82]:


# Make predictions on updated test data
test_preds = ideal_model.predict(df_test)


# We've found an error and it's because our test dataset (after preprocessing) has 101 columns where as, our training dataset (X_train) has 102 columns (after preprocessing).

# In[84]:


# We can find how the columns differ using sets
set(X_train.columns) - set(df_test.columns)


# In this case, it's because the test dataset wasn't missing any 'auctioneerID' fields.
# 
# To fix it, we'll add a column to the test dataset called 'auctioneerID_is_missing' and fill it with False, since none of the 'auctioneerID' fields are missing in the test dataset.

# In[85]:


# Manually adjust df_test to have auctioneerID_is_missing column
df_test["auctioneerID_is_missing"] = False
df_test.head()


# There's one more step we have to do before we can make predictions on the test data.
# 
# And that's to line up the columns (the features) in our test dataset to match the columns in our training dataset.
# 
# As in, the order of the columnns in the training dataset, should match the order of the columns in our test dataset.

# In[86]:


# Match column order from X_train to df_test (to predict on columns, they should be in the same order they were fit on)
df_test = df_test[X_train.columns]


# Now the test dataset column names and column order matches the training dataset, we should be able to make predictions on it using our trained model.

# In[87]:


# Make predictions on the test data
test_preds = ideal_model.predict(df_test)


# In[88]:


test_preds


# In[89]:


# Format predictions into the same format Kaggle is after
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds


# In[90]:


# Export prediction data
df_preds.to_csv("test_predictions.csv", index=False)


# # Feature Importance

# In[91]:


# Find feature importance of our best model
ideal_model.feature_importances_


# In[92]:


# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()


# In[93]:


plot_features(X_train.columns, ideal_model.feature_importances_)


# In[94]:


sum(ideal_model.feature_importances_)


# In[95]:


df.ProductSize.isna().sum()


# In[96]:


df.ProductSize.value_counts()


# In[97]:


df.Turbocharged.value_counts()


# In[98]:


df.Thumb.value_counts()


# In[99]:


df["Enclosure"].value_counts()

