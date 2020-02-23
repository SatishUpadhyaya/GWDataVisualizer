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
                html.H1("GW Data Visualizer", className="display-3"),
                html.P(
                    "GW Data Visualizer allows you to get important analytics and statistics about your dataset.",
                    className="lead",
                ),
                html.Hr(className="my-2"),
                html.P(
                    "About your dataset at the click of a button!", 
                    style={
                        'font-weight': 'bold',
                        'padding-bottom': '10px',
                        'margin': '0 auto', 
                        'display': 'flex', 
                        'align-items': 'center', 
                        'justify-content': 
                        'center'
                    },
                ),
                html.Div(dbc.Button("Generate Charts", color="primary", href="/dataScience"), id='dataScienceStart', style={
                    'padding-top': '10px',
                    'margin': '0 auto', 
                    'display': 'flex', 
                    'align-items': 'center', 
                    'justify-content': 'center',
                }),
            ], 
            className="container"
            ),
    ],
)

showStuff = html.Div(children=[
     html.H2('Here are some quick demos'),
    dcc.Graph(id='example1',
        figure= {
            'data': [
            {'x': [1, 2, 3, 4, 5], 'y': [5, 6, 7, 2, 1], 'type': 'line', 'name':'Type 1'},
            {'x': [1, 2, 3, 4, 5], 'y': [5, 6, 7, 2, 1], 'type': 'bar', 'name':'Type 2'},
            ],
            'layout': {
                'title': 'Basic Histogram and Line Example'
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
     html.Div(dbc.Button("See more", color="primary", href="/examples"), id='exm', style={
                    'padding-top': '10px',
                    'margin': '0 auto', 
                    'display': 'flex', 
                    'align-items': 'center', 
                    'justify-content': 'center',
                }),
])

someStuff = html.Div(children=[
        showStuff
    ],
    className="container"
)

landingPageLayout = html.Div(children=[
    jumbotron,
    someStuff
])