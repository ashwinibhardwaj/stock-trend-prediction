import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


start = '2010-01-01'
end = '2024-05-31'



st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'SBIN.NS')

# Check if user_input is empty
if not user_input:
    st.error('Please enter a stock ticker')
else:
    df = yf.download(user_input, start, end)

    # Describing data
    st.subheader('Data From 2010 - 2024(MAY)')
    st.write(df.describe())

    # VISUALIZATIONS
    st.subheader('Closing Price VS Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, 'b', label='Original')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price VS Time Chart with 100 days Moving Average')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r', label='100MA')
    plt.plot(df.Close, 'b', label='Original')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price VS Time Chart with 200 days Moving Average')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, 'b', label='Original')
    plt.plot(ma100, 'r', label='100MA')
    plt.plot(ma200, 'g', label='200MA')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)


    # splitting data into training and tasting
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    print(data_training.shape)
    print(data_testing.shape)


    scaler = MinMaxScaler(feature_range = (0,1))

    data_training_array = scaler.fit_transform(data_training)


    # loading the model
    model = load_model('keras_model.h5')


    # testing part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform (final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted  = y_predicted*scale_factor
    y_test = y_test*scale_factor

    # final graph
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
