import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
from datetime import datetime

# Set the directory where models are saved
MODEL_SAVE_DIR = "./saved_models"

# Function to load model
def load_model(product_category):
    sanitized_feature = product_category.replace("/", "_").replace("\\", "_")
    model_path = os.path.join(MODEL_SAVE_DIR, f"{sanitized_feature}_prophet_model.pkl")
    
    # Error handling for missing model file
    if not os.path.exists(model_path):
        st.error(f"Model for {product_category} not found.")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load historical sales data (replace with actual data loading process)
df = pd.read_csv('products_data.csv')  
product_categories = ["BEVERAGES", "BREAD/BAKERY", "CLEANING", "DAIRY", "DELI", "GROCERY I", "MEATS", "PERSONAL CARE", "POULTRY", "PRODUCE"]

st.set_page_config(layout="wide")
left_col, right_col = st.columns([1.5, 2]) 

# Add the image to the left column
with left_col:
    st.image("grocery_products.png",) 

# Add forecasting work to the right column
with right_col:

    st.title("Sales Forecasting for Product Categories")

    # Dropdown menu to select product category
    category = st.selectbox("Select Product Category", product_categories)
    model = load_model(category)
    if model is None:
        st.stop() 

    # Prepare the historical data for the selected product
    category_df = df[['date', category]].copy()
    category_df['ds'] = pd.to_datetime(category_df['date'])
    category_df.rename(columns={category: 'y'}, inplace=True)

    # Predict for the future dates (e.g., next 45 days)
    forecast_start_date = datetime(2017, 7, 1)
    prediction_days = 45
    future = model.make_future_dataframe(periods=prediction_days, freq='D')
    forecast = model.predict(future)

    st.subheader("Historical Data with Forecasted Sales")
    fig = go.Figure()

    # Plot the historical data
    fig.add_trace(go.Scatter(x=category_df['ds'], y=category_df['y'], mode='lines', name='Historical Sales', line_shape='spline'))

    # Plot the forecasted data
    forecasted_dates_df = forecast[forecast['ds'] >= forecast_start_date]
    fig.add_trace(go.Scatter(x=forecasted_dates_df['ds'], y=forecasted_dates_df['yhat'], mode='lines', name='Forecasted Sales', line=dict(dash='dash')))

    fig.update_layout(title=f"{category} Sales Forecasting", xaxis_title="Date", yaxis_title="Sales", showlegend=True)
    st.plotly_chart(fig)

    # Custom prediction section
    st.subheader("Get Custom Sales Prediction")
    custom_date = st.date_input("Select a Custom Date", datetime.today())
    if custom_date:
        custom_date = datetime.combine(custom_date, datetime.min.time())  # Convert to datetime.datetime
        
        # Predict sales for the custom date
        custom_forecast = model.predict(pd.DataFrame({'ds': [custom_date]}))
        st.write(f"Predicted Sales for {category} on {custom_date.strftime('%Y-%m-%d')}: {custom_forecast['yhat'][0]:.2f}")
