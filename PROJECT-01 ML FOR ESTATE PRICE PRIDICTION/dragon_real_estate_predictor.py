#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real EState - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("house_data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

##for showing histogram 
#import matplotlib.pyplot as plt
#housing.hist(bins=50, figsize=(20, 15))
# ## Train-Test Splitting

# In[8]:


#for learning purpose we have made it already present in sklearn
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[9]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[10]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(f"ROWS in train set: {len(train_set)}\nROWS in test set: {len(test_set)}")


# In[11]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[12]:


strat_test_set


# In[13]:


strat_test_set.describe()


# In[14]:


strat_test_set.info()


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


# 95/7


# In[18]:


# 376/28


# ## Looking for Correlations

# In[19]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12,8))


# In[21]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


#  ## Trying Out the Attribute Combination

# In[22]:


housing["TAXRM"] = housing["TAX"] / housing["RM"]


# In[23]:


housing.head()


# In[24]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[25]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[26]:


housing = strat_train_set.drop("MEDV", axis=1)   # Features (all columns except MEDV)
housing_labels = strat_train_set["MEDV"].copy()  # Target variable (MEDV column)


# ## Missing Attributes

# In[27]:


# agar data me kuch missing ho to usko thik karne ka method h ye 


# To take care of missing attributes, you have three options:
# 1. Get rid of the missing data points
# 2. Get rid of the whole attribute
# 3. Set the value to some value(0, mean or median)
# 

# In[28]:


a = housing.dropna(subset=["RM"])  # Option 1
# Note that the original housing dataframe will remain unchanged


# In[29]:


a.shape


# In[30]:


housing.drop("RM", axis=1).shape  # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[31]:


median = housing["RM"].median()  # Option 3


# In[32]:


housing["RM"].fillna(median)  # Option 3


# In[33]:


housing.shape


# In[34]:


from sklearn.impute import SimpleImputer
# Create the imputer with median strategy
imputer = SimpleImputer(strategy="median")
# Fit the imputer on the housing dataset
imputer.fit(housing)


# In[35]:


imputer.statistics_


# In[36]:


# Apply the imputer transformation to the housing dataset
X = imputer.transform(housing)

# Convert the transformed data back into a DataFrame with original column names
housing_tr = pd.DataFrame(X, columns=housing.columns)

# Generate descriptive statistics for the transformed DataFrame
housing_tr.describe()


# ## Scikit-learn Design
# 

# Scikit-learn Design
# 
# Primarily, there are three types of objects:
# 
# 1. Estimators:
#    - Estimate some parameter based on a dataset (e.g., Imputer)
#    - Have a fit() method
#    - fit() fits the dataset and calculates internal parameters
# 
# 2. Transformers:
#    - transform() takes input and returns output based on learning from fit()
#    - Also have a convenience method fit_transform()
# 
# 3. Predictors:
#    - Example: LinearRegression
#    - Common methods: fit() and predict()
#    - Also provides score() to evaluate predictions
# 

# ## Creating Pipeline

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[38]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[39]:


housing_num_tr


# ## Selecting a desired model for Dragon Real Estates

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Create the model
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

# Fit the model to training data
model.fit(housing_num_tr, housing_labels)


# In[41]:


some_data = housing.iloc[:5]


# In[42]:


some_labels = housing_labels.iloc[:5]


# In[43]:


prepared_data = my_pipeline.transform(some_data)


# In[44]:


model.predict(prepared_data)


# In[45]:


list(some_labels)


# ## EVALUTATING THE MODEL

# In[46]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[47]:


mse


# ## using better evaluation technique - cross validation

# In[48]:


from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[49]:


rmse_scores


# In[50]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

print_scores(rmse_scores)


# ## SAVING THE MODEL

# In[52]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# ## TESTING THE MODEL ON TEST DATA

# In[57]:


x_test = strat_test_set.drop("MEDV", axis=1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(y_test))


# In[54]:


final_rmse


# In[55]:


final_mse

