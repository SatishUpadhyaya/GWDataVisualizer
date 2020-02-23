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
import numpy as np
from app import app

from plotly.offline import iplot
import plotly.graph_objs as go
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from scipy import stats
import statsmodels.api as sm

import copy
from sklearn.preprocessing import OneHotEncoder

df = None
selectedColumns = None
dropNanChecklist = None
X_train = None
X_test = None
y_test = None
y = None


def backwardSelection(initModel, targCol, pValThres=0.05):
    global X_train
    global X_test
    global y_test
    global y
    modelScores = []
    removedFeatures = [targCol] # of course we want to remove the target
    remainingFeatures = copy.deepcopy(list(X_train.columns)) # initially, we start with all of the features
    currModelScore = initModel.score(X_test, y_test)
    modelScores.append(currModelScore)
    keptFeatures = copy.deepcopy(list(X_train.columns))
    ignoreFeatures = removedFeatures
    
    # first we test to see if initial model contains a feature with a
    # p value that is at least bigger than the pValThres or if 
    # removing the feature with the highest p value improves the model

    largP = float('-inf')
    featWithLargP = None
    nextX = df.drop(columns=removedFeatures)
    featStr = ""
    for c in list(nextX.columns):
        featStr += c + " + "
    featStr = featStr[:-3] # removing the last " + "
    est = ols(formula = targCol + ' ~ ' + featStr, data = df).fit()
    
    for i in range(1, len(est.pvalues)):# we have to start at 1 to avoid the intercept
        nextPVal = est.pvalues[i]
        if nextPVal > largP:
            largP = nextPVal
            
            # we have to subtract 1 since the remainging features 
            # doesn't include the intercept and is thus offset by 1
            # this hinges on the fact that the order of est.pvalues is in the same order as remainingFeatures
            featWithLargP = remainingFeatures[i-1] 
            
    print("feat with larg p is {} + with p of {}".format(featWithLargP, largP))
    remainingFeatures.remove(featWithLargP)
    
    removedFeatures.append(featWithLargP)
#     print(removedFeatures)
    
    nextReg = LinearRegression() 
    X = df.drop(columns = removedFeatures)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    nextReg.fit(X_train, y_train)
    nextModelScore = nextReg.score(X_test,y_test)
    modelScores.append(nextModelScore)
        
    print('Next model score: '+ str(nextModelScore))
    
    # this is the case where removing at least one feature helps improve the model
    if nextModelScore > currModelScore or largP > pValThres:
        reg = nextReg
        keptFeatures = copy.deepcopy(remainingFeatures)
        ignoreFeatures = removedFeatures
        while nextModelScore > currModelScore or largP > pValThres:
            reg = nextReg
            if featWithLargP in remainingFeatures: # handles cases where we remove more than 1 feature
                remainingFeatures.remove(featWithLargP)
            keptFeatures = remainingFeatures
#             print("kp")
#             print(keptFeatures)
            ignoreFeatures = removedFeatures
            currModelScore = nextModelScore
            largP = float('-inf')
            featWithLargP = None
            nextX = df.drop(columns=removedFeatures)
            featStr = ""
            for c in list(nextX.columns):
                featStr += c + " + "
            featStr = featStr[:-3]
            est = ols(formula = targCol + ' ~ ' + featStr, data = df).fit()
            
            for i in range(1, len(est.pvalues)):
                nextPVal = est.pvalues[i]
                if nextPVal > largP:
                    largP = nextPVal
                    # we have to subtract 1 since the remainging features 
                    # doesn't include the intercept and is thus offset by 1
                    featWithLargP = remainingFeatures[i-1] 
            print("HERERRERRER")
            print("feat with larg p is {} + with p of {}".format(featWithLargP, largP))
            
            removedFeatures.append(featWithLargP)
            print(removedFeatures)

            if len(removedFeatures) == len(df.columns):
                return 0, 0, 0, 0

            nextReg = LinearRegression() 
            X = df.drop(columns = removedFeatures)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

            nextReg.fit(X_train, y_train)
            nextModelScore = nextReg.score(X_test,y_test)
            modelScores.append(nextModelScore)

            print('Next model score: '+ str(nextModelScore))
        
        ignoreFeatures.remove(targCol)
        ignoreFeatures.remove(featWithLargP)
        return reg, keptFeatures, ignoreFeatures, modelScores
    # this is the case where removing features from the initial model doesn't help at all
    else: 
        ignoreFeatures = [targCol]
        keptFeatures = copy.deepcopy(list(X_train.columns))
        return initModel, keptFeatures, ignoreFeatures, modelScores



upload = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'padding-bottom': '10px',
            'width': '300px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '0 auto',
            'display': 'block',
            'align': 'center',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])


jumbotron = dbc.Jumbotron(
    [
        html.Div([
                html.H1("GW Data Visualizer", className="display-3", style={
                    'display': 'flex', 
                    'align-items': 'center', 
                    'justify-content': 'center'
                }),
                html.Hr(className="my-2"),
                html.P(
                    "Drag and Drop or Select Files to Continue!", 
                    style={
                        'font-weight': 'bold',
                        'padding-bottom': '10px',
                        'margin': '0 auto', 
                        'display': 'flex', 
                        'align-items': 'center', 
                        'justify-content': 'center'
                    },
                ),
                upload,
            ], 
            className="container"
            ),
    ],
)

def getColIdxStr(inputDF):
    cols = inputDF.columns
    # groups = [i for i in cols if np.issubdtype(inputDF[i].dtype, np.string_)]
    idxs = []
    counter = 0
    for i in cols:
        if np.issubdtype(inputDF[i].dtype, np.string_):
            idxs.append(counter)
        counter += 1
    return idxs


def getDF(filename, contents):
    global df

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            idxs = getColIdxStr(df)
            for i in idxs:
                onehotencoder = OneHotEncoder(categorical_features = [i])
                df = onehotencoder.fit_transform(df).toarray()
            return df
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            idxs = getColIdxStr(df)
            for i in idxs:
                onehotencoder = OneHotEncoder(categorical_features = [i])
                df = onehotencoder.fit_transform(df).toarray()
            return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

def parse_contents(contents, filename, date):
    df = getDF(filename, contents)

    regularComps = html.Div([
        html.Hr(),  # horizontal line
        html.Div([html.P("Select from the dropdown below the things that you want to compare"),], 
        style={
            'padding-top': '10px',
            'margin': '0 auto', 
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center',
        }),
        html.Div([
            dcc.Dropdown(
                id="colsSelected",
                options=[{'label': i, 'value': i} for i in df.columns],
                multi=True, # allow multiple columns to be selected
                searchable=False,
                style={
                    'width': '300px',
                    'margin-right': '20px'
                }
            ),
            dbc.Checklist(
                options=[
                    {"label": "Drop NANs", "value": 1},
                ],
                value=[1],
                id="dropNan",
                switch=True,
            ),
        ],
        style={
            'padding-top': '10px',
            'margin': '0 auto', 
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center',
        }),
        html.Div([
            dbc.Button("Generate Visuals", id="taskButton", color="primary")
        ],
        style={
            'padding-top': '10px',
            'margin': '0 auto', 
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center',
        })
    ])

    if len(df) > 10:
        return html.Div([
            html.H5(filename),

            dash_table.DataTable(
                data=df.to_dict('records')[:10],
                columns=[{'name': i, 'id': i} for i in df.columns]
            ),
            regularComps
        ])
    else: 
        return html.Div([
            html.H5(filename),

            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns]
            ),
            regularComps
        ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(
    Output("output-models", "children"), 
    [Input("taskButton", "n_clicks")],
    [State('colsSelected', 'value'),
     State('dropNan', 'value')])
def on_button_click(n, value, val1):
    global selectedColumns, dropNanChecklist
    selectedColumns = value
    dropNanChecklist = val1
    df.dropna(inplace = True)

    if selectedColumns != None:
        groups = [i for i in selectedColumns if np.issubdtype(df[i].dtype, np.number)]
        boxes = []
        hists = []

        for i in range(len(groups)):
            box = go.Box(
                y = df[groups[i]],
                name = groups[i],
                boxmean = 'sd')
            boxLayout = go.Layout(
                title=go.layout.Title(text='Box Plot of {}'.format(groups[i])),
                xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='Values')),
                yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='Counts'))
            )
            boxes.append(dcc.Graph(id='box{}'.format(i),
                figure= go.Figure(data=[box], layout=boxLayout))
            )

            cnData = df[groups[i]]
            minVal = min(cnData)
            maxVal = max(cnData)
            diff = maxVal - minVal
            hist = go.Histogram(
                x = cnData,
                xbins=dict(
                    size= diff/12
                ),
            )
            histLayout = go.Layout(
                title=go.layout.Title(text='Histogram of {}'.format(groups[i])),
                xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='Values')),
                yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='Counts'))
            )
            hists.append(dcc.Graph(id='hist', 
                figure= go.Figure(data=[hist], layout=histLayout))
            )
        
        pieChart = None
        for cn in selectedColumns:
            columnData = df[cn]
            setLabels = set(columnData)
            listLabels = list(setLabels)
            counterDict = Counter(list(columnData))
            listLabels = []
            listCounts = []
            for key in counterDict:
                listLabels.append(key)
                listCounts.append(counterDict[key])
            pieChart = go.Figure(data=[go.Pie(labels=listLabels, values=listCounts)])

        return html.Div([
            html.Div([i for i in boxes]),
            html.Div([i for i in hists]),
            html.Div(dcc.Graph(figure = pieChart)),
            html.Div([html.H2("Select a target column for Linear Regression"),], 
            style={
                'font-weight': 'bold',
                'padding-top': '20px',
                'margin': '0 auto', 
                'display': 'flex', 
                'align-items': 'center', 
                'justify-content': 'center',
            }),
            html.Div([
            dcc.Dropdown(
                id="targetSelected",
                options=[{'label': i, 'value': i} for i in selectedColumns],
                searchable=False,
                style={
                    'width': '300px',
                    'margin-right': '20px'
                }
            ),
        ],
        style={
            'padding-top': '10px',
            'margin': '0 auto', 
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center',
        }),
        html.Div([
            dbc.Button("Perform Linear Regression", id="mlrButton", color="primary")
        ],
        style={
            'padding-top': '10px',
            'margin': '0 auto', 
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center',
        })
        ])

    else:
        return html.Div([
            dbc.Alert("Please select at least one column", color="danger"),
        ])


@app.callback(
    Output("mlr-output", "children"), 
    [Input("mlrButton", "n_clicks")],
    [State('targetSelected', 'value')])
def on_mlr_button_click(n, value):
    global X_train
    global X_test
    global y_test
    global y

    targetCol = value
    targetCol = targetCol.strip()
    if len(selectedColumns) == 1:
        return html.Div([
            dbc.Alert("Please select multiple items from the first dropdown menu", color="danger")
        ], style={'margin-top': '10px'})
    else:
        if value == None:
            return html.Div([
            dbc.Alert("Please select an item from the first above", color="danger")
            ], style={'margin-top': '10px'})
        else:
            # X = df.drop(columns = targetCol)
            # y = df[targetCol]

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            # initial_X_train = X_train
            # initial_y_train = y_train

            # reg = LinearRegression()
            # reg.fit(initial_X_train, initial_y_train)

            # bestMLRModel, signifFeats, insigniFeats, modScores = backwardSelection(reg, targetCol)
            # if bestMLRModel == 0 and signifFeats == 0 and insigniFeats == 0 and modScores == 0:
            #     return html.Div([
            #         dbc.Alert("There is no correlation between the columns!", color="danger")
            #     ], style={'margin-top': '10px'})

            # return html.Div([
            #     html.H3("Model Score:"),
            #     html.P(str(reg.score(X_test, y_test))),
            #     html.H3("P-Value:"),
            #     html.P(str(pVal)),
            #     html.Div([html.H2("Enter values separated by comma to predict the target value"),], 
            #         style={
            #             'font-weight': 'bold',
            #             'padding-top': '20px',
            #             'margin': '0 auto', 
            #             'display': 'flex', 
            #             'align-items': 'center', 
            #             'justify-content': 'center',
            #         }),
            #         html.Div([
            #             dbc.Input(id="input", placeholder="1, 2, 3", type="text"),
            #         ],
            #         style={
            #             'width': '300px',
            #             'padding-top': '10px',
            #             'margin': '0 auto', 
            #             'display': 'flex', 
            #             'align-items': 'center', 
            #             'justify-content': 'center',
            #         }),
            #         html.Div([
            #             dbc.Button("predict", id="predict", color="primary")
            #         ],
            #         style={
            #             'padding-top': '10px',
            #             'margin': '0 auto', 
            #             'display': 'flex', 
            #             'align-items': 'center', 
            #             'justify-content': 'center',
            #         })
            # ])
            x = df[selectedColumns[0]].values
            y = df[targetCol].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            def linearRegression(a, b):
                Y = a + (b * x)
                return html.Div([
                    dcc.Graph(
                        id='basic-interactions',
                        figure={
                            'data': [
                                {
                                    'x': x,
                                    'y': y,
                                    'name': 'Original Data',
                                    'mode': 'markers',
                                    'marker': {'size': 12}
                                },
                                {
                                    'x': x,
                                    'y': Y,
                                    'name': 'Fitted Data',
                                    'mode': 'line',
                                    'marker': {'size': 12}
                                }
                            ],
                            'layout': {
                                'title': 'Linear Regression',
                                'clickmode': 'event+select'
                            }
                        }
                    ),
                ])

            return linearRegression(intercept, slope)


@app.callback(
    Output("predict-output", "children"), 
    [Input("predict", "n_clicks")],
    [State('input', 'value')])
def on_mlr_button_click(n, value):
    values = value.split(",")
    return html.Div([
        "Value {}".format(values)
    ])


someStuff = html.Div(children=[
    html.Div(id='output-models'),
    html.Div(id='mlr-output'),
    html.Div(id='predict-output')
],
className='container')

dataSciencePagelayout = html.Div(children =[
    jumbotron, 
    someStuff
])