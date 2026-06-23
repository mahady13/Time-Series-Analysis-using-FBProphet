import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly,plot_components
import plotly

df=pd.read_csv("BTC-USD.csv")
st.title('Time Series Analysis using FBProphet')
st.header('Bitcoin Price Prediction using Dataset ranging from 2014-2022')
st.sidebar.header("Click To See The Step By Step Process")
selected_button=st.sidebar.radio('Select an Option',['Primary Dataset','Preprocessed Data','BTC Price Graph','Forecasted Graph','Seasonal Components'])

if selected_button=='Primary Dataset':
    st.header("BTC Price Dataset before Preprocessing")
    st.dataframe(df)

elif selected_button=='Preprocessed Data':
    df=df.rename(columns={'Date':'ds','Close':'y'})
    df['ds']=pd.to_datetime(df['ds'])
    df['y']=pd.to_numeric(df['y'])
    st.header("BTC Price Dataset after Preprocessing")
    st.dataframe(df)

elif selected_button=='BTC Price Graph':
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])
    st.header("BTC Price Graph")
    st.line_chart(data=df,x='ds',y='y')

elif selected_button=='Forecasted Graph':
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])
    m=Prophet()
    model=m.fit(df)
    future=m.make_future_dataframe(periods=1700,freq='D')
    forecast=m.predict(future)
    st.header('BTC Price Future Forecast Graph (Interactive)')
    fig=m.plot(forecast)
    st.plotly_chart(fig,use_container_width=True)

elif selected_button=='Seasonal Components':
    df = pd.read_csv("BTC-USD.csv")
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])
    st.header('BTC Price Seasonality Components')
    m = Prophet()
    model = m.fit(df)
    future = m.make_future_dataframe(periods=1700, freq='D')
    forecast = m.predict(future)
    fig2=m.plot_components(forecast)
    st.pyplot(fig2,use_container_width=True)




