import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from pandas.plotting import autocorrelation_plot, lag_plot
from operator import itemgetter
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

def time_series_3d():
    #greatest_variance = ['MS_winds.mat', 'charlotte.mat', 'hawaii_3.mat', 'india_1.mat', 'pacific_2.mat']
    greatest_variance = ['charlotte.mat']
    for file in greatest_variance:
        print(file)
        og_file = 'data/'+file
        og_data = pd.read_csv(og_file, header=None)
        base = datetime.datetime(2014, 1, 1, 0, 0)
        date_list = [base + datetime.timedelta(hours=x) for x in range(og_data.shape[0])]
        # og_data['date'] = date_list
        og_data = og_data.iloc[:40, :5]
        test_data = og_data.iloc[-150:, :10]
        ax = plt.figure().add_subplot(projection='3d')

        # Plot a sin curve using the x and y axes.
        x = og_data.index
        x_pasthistory = x[:23]
        x_input = x[22:-5]
        x_predicted = x[-6:]

        for col in og_data.columns:
            y = og_data[col]

            y_pasthistory = y[:23]
            y_input = y[22:-5]
            y_predicted = y[-6:]

            ax.plot(x_pasthistory, y_pasthistory, zs=col, zdir='y', label=col, color='green')
            ax.plot(x_input, y_input, zs=col, zdir='y', label=col, color='blue')
            ax.plot(x_predicted, y_predicted, zs=col, zdir='y', label=col, color='red')


        # Make legend, set axes limits and labels
        green_patch = mpatches.Patch(color='green', label='Past history')
        blue_patch = mpatches.Patch(color='blue', label='Input horizon')
        red_patch = mpatches.Patch(color='red', label='Forecast horizon', linewidth=0.1)
        ax.legend(handles=[red_patch, blue_patch, green_patch])
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.set_zlim(0, 1)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Station')
        ax.set_zlabel('Wind speed (m/s)')

        ax.view_init(elev=20., azim=-35)

        plt.show()


def spider_plot():
    df = pd.read_csv('output/mae_stations_def.csv', sep='&')
    import plotly.graph_objects as go
    import plotly.offline as pyo


    categories = df['dataset']
    categories = [*categories, categories[0]]

    mae = df['mae']
    mae_stacked = df['mae_stacked']
    
    mae = [*mae, mae[0]]
    mae_stacked = [*mae_stacked, mae_stacked[0]]


    fig = go.Figure(
        data=[
            go.Scatterpolar(r=mae, theta=categories, fill='toself', name='Base model 1 MAE'),
            go.Scatterpolar(r=mae_stacked, theta=categories, fill='toself', name='Ensemble MAE'),
            
        ],
        layout=go.Layout(
            title=go.layout.Title(text='MAE comparison'),
            polar={'radialaxis': {'visible': True}},
            showlegend=True
        )
    )

    pyo.plot(fig)

time_series_3d()
spider_plot()