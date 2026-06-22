import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly,plot_components

df=pd.read_csv("BTC-USD.csv")
st.title('Time Series Analysis using FBProphet')
st.header('Bitcoin Price Prediction using Dataset ranging from 2014-2022')
st.header("BTC Price Dataset before Preprocessing")
st.dataframe(df)

df=df.rename(columns={'Date':'ds','Close':'y'})
df['ds']=pd.to_datetime(df['ds'])
df['y']=pd.to_numeric(df['y'])
st.header("BTC Price Dataset after Preprocessing")
st.dataframe(df)

st.header("BTC Price Graph")
st.line_chart(data=df,x='ds',y='y')


m=Prophet()
model=m.fit(df)
future=m.make_future_dataframe(periods=1700,freq='D')
forecast=m.predict(future)
st.header('Dataset after using FBProphet')
st.dataframe(forecast)
fig=m.plot(forecast)
st.pyplot(fig)




