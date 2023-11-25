import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from src import utils
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# -------------------------------------------
# Load Function
# -------------------------------------------
# Isinya sih gak bagus ya, jadi jangan ditiru
def read_data(data_path):
    """Read the history data"""
    # Read & parse data
    data = pd.read_csv(data_path, parse_dates=['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    return data

def find_outlier(data, lb=None, ub=None, type='constant'):
    """Cari UB dan LB dari data"""
    if type=='constant':
        pass
    else:
        q1, q3 = np.quantile(data, q=[0.25, 0.75])
        iqr = q3-q1
        lb = q1 - 1.5*iqr
        ub = q3 + 1.5*iqr

    return lb, ub

def is_anomaly(data, lb, ub):
    if (data>ub) or (data<lb):
        return 1
    else:
        return 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_green(x, ub):
    demeaned_x = x - ub
    return sigmoid(demeaned_x/10_000)

def sigmoid_red(x, lb):
    demeaned_x = x - lb
    return sigmoid(-demeaned_x/10_000)

def proba_anomaly_forecast(data, bs):
    return sigmoid_green(data, bs[-2]) + sigmoid_red(data, bs[1])

# -------------------------------------------
# Data Preparation
# -------------------------------------------
# Baca semua data
data_path = 'data/sales_history.csv'
actual_data = read_data(data_path)

# Tentukan current data
current_date = '2011-07-08'
cond_train = actual_data['Date'] <= current_date
current_data = actual_data[cond_train]

# Tentukan data anomaly {0: tidak anomali, 1: anomali}
# Asumsi semua data adalah tidak anomali
current_data['is_anomaly'] = 0


# -------------------------------------------
# Model Preparation
# -------------------------------------------
# Load forecasting model
model = utils.pickle_load('artefacts/best_model_july.pkl')

# Cari outlier (untuk anomaly detection)
lb, ub = find_outlier(
    data=current_data['Weekly_Sales'],
    lb = 15_000,
    ub = 50_000,
    type = 'constant'
)
bs = np.linspace(lb, ub, 5)


# -------------------------------------------
# Dashboard Preparation
# -------------------------------------------
st.set_page_config(
    page_title = 'Near Real-Time Machine Monitoring Dashboard',
    layout = 'wide'
)
st.title('MachineID: XML-0013')
st.divider()

# Set placeholder untuk update graph
placeholder_1 = st.empty()
placeholder_2 = st.empty()


# -------------------------------------------
# Monitoring Preparation
# -------------------------------------------
# set index terakhir dari current_data
i = current_data.index[-1]

# Buat placeholder untuk forcasted data
context_window = 5
col_to_forecast = ['Year', 'Month', 'Day']
forecasted_data = actual_data.loc[[i+future for future in range(context_window)]]
forecasted_data['Weekly_Sales'] = model.predict(
    forecasted_data[col_to_forecast]
)
forecasted_data['is_anomaly'] = forecasted_data['Weekly_Sales'].apply(lambda x: is_anomaly(x, lb, ub))
forecasted_data['proba_anomaly'] = forecasted_data['Weekly_Sales'].apply(lambda x: proba_anomaly_forecast(x, bs))
# -------------------------------------------
# Start Monitoring
# -------------------------------------------
n_actual = len(actual_data)
for seconds in range(n_actual-i-context_window):
    # DATA AKTUAL & INFORMASI ANOMALI / TIDAK
    # -------------------------------------------
    # Definisikan apakah data sales sekarang anomaly/tidak
    current_sales = actual_data['Weekly_Sales'].loc[i+1]
    anomaly_status = is_anomaly(current_sales, lb, ub)

    # Masukkan data baru ke dalam data current
    new_data = actual_data.loc[[i+1]].copy()
    new_data['is_anomaly'] = anomaly_status
    current_data = pd.concat([current_data, new_data], axis=0)
    

    # DATA FORECASTING & INFORMASI ANOMALI / TIDAK
    # -------------------------------------------
    x = forecasted_data[col_to_forecast].iloc[[-1]]
    pred = model.predict(x)[0]
    forecasted_data['Weekly_Sales'].iloc[-1] = pred
    forecasted_data['is_anomaly'] = forecasted_data['Weekly_Sales'].apply(lambda x: is_anomaly(x, lb, ub))
    forecasted_data['proba_anomaly'] = forecasted_data['Weekly_Sales'].apply(lambda x: proba_anomaly_forecast(x, bs))

    # Show KPI
    with placeholder_1.container():
        kpi1, kpi2, kpi3 = st.columns(3)

        # KPI 1 - n anomaly
        n_anomaly = current_data[current_data['is_anomaly']==1].shape[0]
        n_data = current_data.shape[0]
        kpi1.metric(label="**Number of anomaly**", 
                    value=f"{n_anomaly} / {n_data}")

        # KPI 2 - % anomaly
        pct_anomaly = np.round((n_anomaly/n_data) * 100, 2)
        kpi2.metric(label="**Percent of anomaly**", 
                    value=f"{pct_anomaly} %")

        # KPI 2 - % Forcasted probability
        proba_anomaly = forecasted_data['proba_anomaly'].iloc[-1]
        kpi3.metric(label="**Probability of anomaly**", 
                    value=f"{proba_anomaly:.2f}")
            
    # Show live-time
    with placeholder_2.container():
        # Trace data aktual
        trace_actual = go.Scatter(
            x=current_data['Date'],
            y=current_data['Weekly_Sales'],
            mode='lines',
            name=' actual data',
            line=dict(width=2)
        )

        # Trace data akutal yang anomali
        trace_actual_anomaly = go.Scatter(
            x=current_data['Date'][current_data['is_anomaly']==1],
            y=current_data['Weekly_Sales'][current_data['is_anomaly']==1],
            mode='markers',
            name=' anomalous data',
            marker = dict(color='firebrick', size=10, opacity=0.4)
        )

        # Trace anomaly
        trace_anomaly_lb = go.Scatter(
            x=current_data['Date'],
            y=np.ones(len(current_data)) * lb,
            mode='lines',
            showlegend=False,
            line=dict(color='grey', width=0.5, dash='dash')
        )

        trace_anomaly_ub = go.Scatter(
            x=current_data['Date'],
            y=np.ones(len(current_data)) * ub,
            mode='lines',
            showlegend=False,
            line=dict(color='grey', width=0.5, dash='dash')
        )

        # Trace Forecast
        trace_forecast = go.Scatter(
            x=forecasted_data['Date'],
            y=forecasted_data['Weekly_Sales'],
            mode='lines',
            name=' forecasted data',
            line=dict(width=2, color='orange', dash='dash')
        )

        trace_forecast_anomaly = go.Scatter(
            x=forecasted_data['Date'][forecasted_data['is_anomaly']==1],
            y=forecasted_data['Weekly_Sales'][forecasted_data['is_anomaly']==1],
            mode='markers',
            name=' forecasted anomalous data',
            marker = dict(color='black', size=6, opacity=0.4),
            marker_symbol='x'
        )

        # Summarize data
        trace_all = [trace_actual, 
                     trace_actual_anomaly,
                     trace_anomaly_lb, trace_anomaly_ub,
                     trace_forecast,
                     trace_forecast_anomaly]

        # Plot figure
        layout = go.Layout(
            # autosize=True,
            # width=1800,
            height=250,
            xaxis_title='Period',
            yaxis_title='Load [Pa]',
            # legend=dict(font_size=20),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig = go.Figure(data=trace_all, layout=layout)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        time.sleep(0.75)


    # add new on df_predict
    i+=1

    # Update the forecasted data
    new_data_ = actual_data.loc[[i+context_window-1]]
    forecasted_data = pd.concat([forecasted_data, new_data_])