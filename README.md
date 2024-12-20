# Product Sales Forecasting Using Prophet

## **Overview**
This project leverages advanced time series forecasting techniques to predict sales for various product categories. By using historical sales data, the goal is to enable businesses to make data-driven decisions to optimize inventory, improve resource allocation, and maximize revenue.


## Live Demo
[Product Category Sales Forecasting App](https://appforecasting.streamlit.app/)

## **Key Features**
- Forecast sales for 10 different product categories.
- Historical data exploration and preprocessing.
- Evaluation of ARIMA and Prophet models for forecasting.
- Deployment of a user-friendly Streamlit app for interactive forecasting.

## **Dataset**
The dataset, sourced from Kaggle, contains sales data from 2013 to 2017, including:
- **date**: Transaction date
- **store_nbr**: Store identifier
- **product_category**: Product classification
- **sales**: Number of units sold
- **onpromotion**: Number of items on promotion

### **Data Preprocessing**
- Filtered data from 2015 onwards for analysis.
- Selected 10 key product categories.
- Aggregated and cleaned the data for model training.

## **Models Used**
1. **ARIMA**
   - Parameter tuning: Explored combinations of (p, d, q).
   - Results: 
     - PRODUCE: MAPE = 10.48%
     - DAIRY: MAPE = 14.60%

2. **Prophet**
   - Fine-tuned parameters:
     - `changepoint_prior_scale`
     - `seasonality_prior_scale`
   - Results:
     - PRODUCE: MAPE = 5.39%
     - DAIRY: MAPE = 6.48%

   **Conclusion**: Prophet outperformed ARIMA in both accuracy and interpretability.

## **Streamlit Application**
The interactive app allows users to:
- View historical sales data alongside future forecasts.
- Select specific product categories for visualization.
- Obtain custom sales predictions for selected dates.

## **Limitations**
- Limited to 10 product categories for simplicity.
- High dependency on the quality and availability of historical data.

## **Future Work**
- Expand to include more product categories and stores.
- Incorporate external factors like weather, holidays, and promotions.
- Use additional models for comparison (e.g., LSTM, XGBoost).

## **Technologies Used**
- **Python**: Data manipulation and model development.
- **Prophet & ARIMA**: Forecasting models.
- **Streamlit**: Web application framework.
- **Pandas, NumPy, Matplotlib**: Data analysis and visualization.

