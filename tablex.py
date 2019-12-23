import ipdb

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('qx.csv')

COL1 = 'Sex'
COL2 = 'UNCD'

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in df[COL1].unique()],
                value='M',
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Surface', 'Heatmap']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in df[COL2].unique()],
                value='PB',
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.RangeSlider(
        id='crossfilter-year--slider',
        min=df['Issue Age'].min(),
        max=df['Issue Age'].max(),
        value=[df['Issue Age'].min(), df['Issue Age'].max()],
        #marks={str(year): str(year) for year in df['Issue Age'].unique()},
        step=None
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Issue Age'].between(*year_value)]
    dfp = dff.set_index(['Sex', 'UNCD', 'Issue Age'])
    dfp = dfp.pivot(columns='Duration')
    dfp = dfp.mean(level=-1)['qx'] # average over all but the last level (all features)
    if True:
        #X, Y = pd.np.meshgrid(dfp.index, dfp.columns)
        this_graph = go.Surface(
            z=dfp.values,
            x=dfp.index,
            y=dfp.columns
        )
    else:
        this_graph = dict(
            x=dff[dff[COL1] == xaxis_column_name]['Issue Age'],
            y=dff[dff[COL2] == yaxis_column_name]['qx'],
            text=dff[dff[COL2] == yaxis_column_name]['qx'],
            customdata=dff[dff[COL2] == yaxis_column_name]['qx'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )
    return {
        'data': [this_graph],
        'layout': dict(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


def create_time_series(x, y, axis_type, title):
    return {
        'data': [dict(
            x=x,
            y=y,
            mode='lines+markers'
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis': {'showgrid': False}
        }
    }


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    print("UYT", xaxis_column_name, axis_type, hoverData)
    #ipdb.set_trace()
    this_x = hoverData['points'][0].get('x')
    if this_x is not None:
        dff = df[df['Issue Age'] == this_x]
        dfp = dff.set_index(['Sex', 'UNCD', 'Duration'])
        dfp = dfp.pivot(columns='Issue Age')
        dfp = dfp.mean(level=-1) 
        title = ''
        #title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
        return create_time_series(dfp.index, dfp.unstack().values, axis_type, title)
    else:
        return {'data': []}


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    print("UXT", yaxis_column_name, axis_type, hoverData)
    this_y = hoverData['points'][0].get('y')
    if this_y is not None:
        dff = df[df['Duration'] == this_y]
        dfp = dff.set_index(['Sex', 'UNCD', 'Issue Age'])
        dfp = dfp.pivot(columns='Duration')
        dfp = dfp.mean(level=-1) 
        return create_time_series(dfp.index, dfp.unstack().values, axis_type, yaxis_column_name)
    else:
        return {'data': []}

                 

if __name__ == '__main__':
    app.run_server(debug=True)

