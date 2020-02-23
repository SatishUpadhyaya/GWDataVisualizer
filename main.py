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

from landingPage import landingPageLayout
from dataSciencePage import dataSciencePagelayout
from examplesPage import examplesPagelayout
from app import app 


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/index")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Data Visualizer", href="/dataScience"),
                dbc.DropdownMenuItem("Examples", href="/examples"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="GW Data Visualizer",
    brand_href="/index",
    color="primary",
    dark=True,
)

footer = html.Div(html.Div(children=[
    html.A([html.Img(src=app.get_asset_url('facebook.png'), style={'padding-top': '14px', 'padding-right': '14px', 'height': '34px'})], href="https://facebook.com"),
    html.A([html.Img(src=app.get_asset_url('linkedin.png'), style={'padding-top': '14px', 'padding-right': '14px', 'height': '34px'})], href="https://linkedin.com"),
    html.A([html.Img(src=app.get_asset_url('github.png'), style={'padding-top': '10px', 'padding-right': '14px', 'height': '34px'})], href="https://github.com"),
    html.P(
        "2020 GW Data Visualizer. All rights reserved. ", 
        style={
            'padding-top': '10px',
            'margin': '0 auto', 
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center'
        },
    ),
]), style={
        'margin-top': '30px',
        'width': '100%',
        'background-color': '#007bff',
        'color': 'white',
        'text-align': 'center',
        'height': '80px',
        'padding-bottom': '0', 
        'margin-bottom': '0',
        'bottom': '0'
    })


app.layout = html.Div([
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    footer
])


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dataScience':
        return dataSciencePagelayout
    elif pathname == '/examples':
        return examplesPagelayout
    else:
        return landingPageLayout


if __name__ == '__main__':
    app.run_server(port=8080, debug= True)
