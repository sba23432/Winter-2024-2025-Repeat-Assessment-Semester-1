#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import streamlit as st


# In[2]:


##Load data
final_df = pd.read_csv("final_df.csv")


# In[3]:


# Preprocess the dataset
#Date must be index for timeseries
final_df['date'] = pd.to_datetime(final_df['date'])
final_df.set_index('date', inplace=True)
final_df = final_df.sort_index()


# In[4]:


y = final_df['num_rentals']


# In[5]:


# Split into training and test sets
#80% of rental data taken
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:] ##tales the remaining 20%


# In[6]:


# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # Adjust the order based on diagnostics
fitted_model = model.fit()


# In[7]:


# Forecast
forecast = fitted_model.forecast(steps=len(test))


# In[8]:


# Evaluate the model
mae = mean_absolute_error(test, forecast)
print(f"Mean Absolute Error: {mae}")


# In[9]:


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data', color='orange')
plt.plot(test.index, forecast, label='Forecast', color='green')
plt.legend()
plt.title("ARIMA Forecast for Number of Rentals")
plt.show()


# In[10]:


#Display MAE
st.markdown(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.markdown("The MAE measures the forecast's accuracy. A lower MAE indicates a better fit.")

# **Weather Analysis Module**
st.subheader("Weather Analysis")
st.markdown("Explore how weather conditions impact cycling activity.")

# Scatter Plot for Weather vs Rentals
weather_metric = st.selectbox("Select Weather Metric to Analyze:", ["maxtp", "mintp", "rain", "wdsp"])
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(final_df[weather_metric], final_df['num_rentals'], alpha=0.5, color='blue')
ax2.set_title(f"Number of Rentals vs {weather_metric.capitalize()}")
ax2.set_xlabel(weather_metric.capitalize())
ax2.set_ylabel("Number of Rentals")
st.pyplot(fig2)


# In[ ]:





# In[ ]:




