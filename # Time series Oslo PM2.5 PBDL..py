#!/usr/bin/env python
# coding: utf-8

# # PBDL Oslo PM2.5

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from scipy.stats import randint as sp_randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error



# Define a function to create datasets with varying time steps
def create_time_series_dataset(X, y, time_steps, forecast_days):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - forecast_days + 1):
        Xs.append(X[i:i + time_steps].flatten())  # Flatten the time steps dimension
        ys.append(y.iloc[i + time_steps:i + time_steps + forecast_days].values)
    return np.array(Xs), np.array(ys)
# Loop to evaluate models with different time steps and forecast days
for time_steps in [1, 7, 10]:  # Time steps
    for forecast_days in [1, 7, 10]:  # Forecast for 1, 7, and 10 days
        print(f"\nEvaluating models with Time Steps: {time_steps}, Forecast Days: {forecast_days}")
        mae, rmse, feature_importance_df = tune_and_train(time_steps, forecast_days)
        results[(time_steps, forecast_days)] = {'mae': mae, 'rmse': rmse, 'feature_importance': feature_importance_df}


# Define hyperparameters for random search
param_dist = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'num_units': sp_randint(50, 200),
    'num_layers': sp_randint(1, 4),
    'kernel_regularizer': [None, l2(0.01), l2(0.001)]
}

# Function to perform hyperparameter tuning and training
def tune_and_train(time_steps, forecast_days):
    X_ts, y_ts = create_time_series_dataset(pd.DataFrame(X_scaled), y, time_steps, forecast_days)    

# Define a function to create datasets with varying time steps
def create_time_series_dataset(X, y, time_steps, forecast_days):
    X = X.to_numpy()  # Convert DataFrame to NumPy array
    Xs, ys = [], []
    for i in range(len(X) - time_steps - forecast_days + 1):
        Xs.append(X[i:i + time_steps].flatten())  # Flatten the time steps dimension
        ys.append(y.iloc[i + time_steps:i + time_steps + forecast_days].values)
    return np.array(Xs), np.array(ys)
# Load data
df = pd.read_excel("PM25OsloTS.xlsx")

# Temporal feature processing
df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y')
df['Year'] = df['Time'].dt.year
df['Month'] = df['Time'].dt.month
df['Day'] = df['Time'].dt.day
df = df.sort_values(by='Time')

# Extract features and target variable
X = df[["TV", "Tmean", "HDD", "VP", "WS", "WG", "meanRH", "SD", "PP", "Year", "Month","Day"]]
y = df["PM25"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training, staging, validation, and test sets
    X_train, X_staging, y_train, y_staging = train_test_split(X_ts, y_ts, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_staging, y_staging, test_size=0.5, random_state=42)


# Define the ODE function (using y'' + y = 0 )
def ode_function(y_pred):
    # Assume y_pred is the model's predicted output and apply a simple ODE constraint y'' + y = 0
    # Here, we approximately compute the second derivative of y_pred and return the ODE constraint
    y_pred_diff2 = tf.gradients(y_pred, [time_steps])[0]  # Compute the second derivative
    ode_constraint = y_pred_diff2 + y_pred  # ODE constraint: y'' + y = 0
    return ode_constraint



# Custom layer to incorporate ODE function
class ODELayer(tf.keras.layers.Layer):
    def __init__(self, ode_function, **kwargs):
        super(ODELayer, self).__init__(**kwargs)
        self.ode_function = ode_function

    def call(self, inputs):
        # Apply the ODE function,
        return self.ode_function(*inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]  # Output shape same as input shape except the last dimension  

# Define a physical-based loss function with L2 regularization
def ode_system(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Example: Mean Squared Error
    return loss

import tensorflow as tf

# Define the loss function with an ODE constraint
def custom_loss(y_true, y_pred, ode_loss_weight=0.1):
    """
    Custom loss function combining mean squared error (MSE) with an ODE constraint.
    Args:
        y_true: True target values.
        y_pred: Predicted values from the model.
        ode_loss_weight: Weighting factor for the ODE constraint loss.
    Returns:
        A combined loss value.
    """
    # Compute the standard data loss (Mean Squared Error)
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # Compute the ODE constraint loss (assuming y'' + y = 0)
    with tf.GradientTape() as tape1:
        tape1.watch(y_pred)
        first_derivative = tape1.gradient(y_pred, time_steps)  # Compute the first derivative

    with tf.GradientTape() as tape2:
        tape2.watch(first_derivative)
        second_derivative = tape2.gradient(first_derivative, time_steps)  # Compute the second derivative

    ode_loss = tf.reduce_mean(tf.square(second_derivative + y_pred))  # Enforce y'' + y = 0

    # Combine data loss and ODE loss
    total_loss = data_loss + ode_loss_weight * ode_loss

    return total_loss


# Define custom estimator class
class CustomEstimator(BaseEstimator):
    def __init__(self, learning_rate, num_units, num_layers, kernel_regularizer=None):
        self.learning_rate = learning_rate
        self.num_units = num_units
        self.num_layers = num_layers
        self.kernel_regularizer = kernel_regularizer

        
    def fit(self, X, y):
        model = Sequential()
        model.add(Dense(units=self.num_units, kernel_initializer="he_normal", input_shape=(12,), kernel_regularizer=self.kernel_regularizer))
        model.add(BatchNormalization())
        model.add(Activation("elu"))
        for _ in range(self.num_layers - 1):
            model.add(Dense(units=self.num_units, kernel_initializer="he_normal", kernel_regularizer=self.kernel_regularizer))
            model.add(BatchNormalization())
            model.add(Activation("elu"))
        model.add(Dense(units=1))
        optimizer = tf.keras.optimizers.Nadam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=ode_system, metrics=['mse'])
        self.model = model
        self.history = self.model.fit(X, y, epochs=1000, batch_size=32, verbose=0)
        return self

# Define hyperparameters for random search
param_dist = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'num_units': sp_randint(50, 200),
    'num_layers': sp_randint(1, 4),
    'kernel_regularizer': [None, l2(0.01), l2(0.001)]
}

# Create an instance of CustomEstimator
custom_estimator_instance = CustomEstimator(learning_rate=0.001, num_units=100, num_layers=2)

# Define custom scorer
def custom_scorer(estimator, X, y):
    return -estimator.model.evaluate(X, y, verbose=0)[0]  # Return negative validation loss

# Create and fit RandomizedSearchCV with the instance of CustomEstimator
random_search = RandomizedSearchCV(estimator=custom_estimator_instance, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, random_state=42, scoring=custom_scorer)
random_search_result = random_search.fit(X_train, y_train)
 
# Print best parameters and best score
print("Best Parameters: ", random_search_result.best_params_)
print("Best Score: ", random_search_result.best_score_)

# Access the best model from the RandomizedSearchCV
best_model = random_search.best_estimator_.model


    # Train the model with best parameters
    history = best_model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), verbose=1)
 



  


# In[6]:


from sklearn.metrics import mean_squared_error

# Make predictions on the validation set
y_pred = best_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)


# In[8]:


best_learning_rate = best_model.optimizer.learning_rate.numpy()
best_regularizer = best_model.layers[0].kernel_regularizer.get_config()['l2']

print("Best Learning Rate:", best_learning_rate)
print("Best L2 Regularizer:", best_regularizer)


# In[13]:


# Make predictions on the validation set

y_pred = model.predict(X_val)


# Plot predicted vs. actual values
plt.figure(figsize=(8, 6))
plt.plot(y_val, label='Actual')
plt.plot(y_val, label='Predicted')
plt.xlabel('Actual PM2.5 values')
plt.ylabel('Predicted PM2.5 values')
plt.title('Actual vs. Predicted Oslo PM2.5 values based on PBDL')
plt.legend()
plt.show()


# In[14]:


# 对损失进行可视化
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('The loss curves of the PBDL for the training set and the test set (The PM2.5 levels in Oslo)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', style='italic', loc='bottom')
plt.legend()
plt.show()





# In[ ]:




