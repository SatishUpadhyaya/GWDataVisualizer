import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_table

import base64
import datetime
import io
import pandas as pd


jumbotron = dbc.Jumbotron(
    [
        html.Div([
                html.H1("Quick Demo", className="display-3", style={
                    'display': 'flex', 
                    'align-items': 'center', 
                    'justify-content': 'center'
                }),
                html.Hr(className="my-2"),
                html.P(
                    "Here are some quick demos about what you can do with GW Data Visualizer", 
                    style={
                        'padding-bottom': '10px',
                        'margin': '0 auto', 
                        'display': 'flex', 
                        'align-items': 'center', 
                        'justify-content': 'center'
                    },
                ),
            ], 
            className="container"
            ),
    ],
)

showStuff = html.Div(children=[
    dcc.Graph(id='example1',
        figure= {
            'data': [
            {'x': [1, 2, 3, 4, 5], 'y': [5, 6, 7, 2, 1], 'type': 'bar', 'name':'Type 1'},
            {'x': [2, 4, 6, 8, 10], 'y': [10, 12, 14, 4, 2], 'type': 'bar', 'name':'Type 2'},
            ],
            'layout': {
                'title': 'Basic Bar Chart One Example'
            }
        }),
        dcc.Graph(id='example1',
        figure= {
            'data': [
            {'x': [1, 2, 3, 4, 5], 'y': [5, 6, 7, 2, 1], 'type': 'line', 'name':'One Type'},
            ],
            'layout': {
                'title': 'Basic Line Graph Example'
            }
        }),
        dcc.Graph(
        id='basic-interactions',
        figure={
            'data': [
                {
                    'x': [1, 2, 3, 4],
                    'y': [4, 1, 3, 5],
                    'text': ['a', 'b', 'c', 'd'],
                    'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                    'name': 'Type 1',
                    'mode': 'markers',
                    'marker': {'size': 12}
                },
                {
                    'x': [1, 2, 3, 4],
                    'y': [9, 4, 1, 4],
                    'text': ['w', 'x', 'y', 'z'],
                    'customdata': ['c.w', 'c.x', 'c.y', 'c.z'],
                    'name': 'Type 2',
                    'mode': 'markers',
                    'marker': {'size': 12}
                }
            ],
            'layout': {
                'title': 'Basic Graph Interaction',
                'clickmode': 'event+select'
            }
        }
    ),
])

someStuff = html.Div(children=[
    showStuff
],
className='container')

examplesPagelayout = html.Div(children =[
    jumbotron, 
    someStuff
])