import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
from stocknews import StockNews
#import plotly.express as px
import plotly.graph_objects as go
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly

start = st.sidebar.date_input('Start Date', datetime.date(2010, 1, 1))
#end = '2023-01-01'
#end = date.today()
end = st.sidebar.date_input('End Date')

#@st.cache
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker' , 'AAPL')
#df = data.DataReader('AAPL', 'yahoo' , start , end)
df = yf.download(user_input, start=start, end=end)

moving_average , testing ,forecasting ,news = st.tabs(['Moving Average','Testing','Forecasting','News'])

with moving_average:
    st.write('Price')

    #DESCRIBING DATA
    st.subheader('Data from 2010 - 2023')
    st.write(df.describe())

    #data visualization
    st.subheader('Closing Price vs Time')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
    st.plotly_chart(fig)

    st.subheader('Closing Price vs Time with 100MA')
    ma100 = df['Close'].rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100-day Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))

    fig.update_layout(title='Closing Price vs Time with 100MA',
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)


    st.subheader('Closing Price vs Time with 100MA & 200MA')
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100-day Moving Average', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200-day Moving Average', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='green')))
    
    fig.update_layout(title='Closing Price vs Time with 100MA & 200MA',
                    xaxis_title='Date',
                    yaxis_title='Price')
    st.plotly_chart(fig)

    #Splitting data for training and testing

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0,1))

    data_training_array = scaler.fit_transform(data_training)

    #load my model
    model = load_model('keras_model.h5')

    #testing part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append((input_data[i-100: i]))
        y_test.append(input_data[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_predicted = model.predict(x_test)

    #for factor by which data has been scaled down
    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor


    #final graph


with testing:
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test , 'b' , label= 'Original price')
    plt.plot(y_predicted , 'r' , label= 'Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    st.write('Prediction')
    st.subheader('Predictions vs Original')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=y_test, mode='lines', name='Original price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=y_predicted.flatten(), mode='lines', name='Predicted price', line=dict(color='red')))

    fig.update_layout(title='Predictions vs Original',
                    xaxis_title='Time',
                    yaxis_title='Price')
    st.plotly_chart(fig)

with forecasting:
    data = yf.download(user_input, start=start, end=end)
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365
    
    # Resetting the index to get the date as a column
    df_train = data.reset_index()[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


with news:
    st.write('News')
    st.subheader(f'News of {user_input}')
    sn = StockNews(user_input , save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News sentiment {news_sentiment}')
