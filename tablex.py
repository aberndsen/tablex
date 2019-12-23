import ipdb
import argparse
import copy
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

import plotly.graph_objects as go

import dash_reusables as dr

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

#
# Number of columns that are not Issue Age (xaxis), Duration (yaxis) or qx (zaxis)
#
MAX_FEATURE_COLS = 6

MAIN_HEIGHT = 600

#
# Attained Age alias columns, otherwise calculate
# use lower case
AA_ALIASES = ['attained age', 'aa', 'atta', 'atta_age']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

df = None
aa_col = None
xaxis = None
yaxis = None
zaxis = None
feature_cols = []


def load_and_filter_data(
        col0_filter=None,
        col1_filter=None,
        col2_filter=None,
        col3_filter=None,
        col4_filter=None,
        col5_filter=None,
        x_val=None,
        y_val=None):
    dff = df
    for feat_num, feat_filter in enumerate(
            [col0_filter, col1_filter, col2_filter, col3_filter, col4_filter, col5_filter]):
        if feat_filter is not None:
            if not isinstance(feat_filter, (list, tuple)):
                feat_filter = [feat_filter]
            if len(feat_filter):
                keepers = dff[feature_cols[feat_num]].isin(feat_filter)
                dff = dff[keepers]

    if dff is None:
        dff = df

    if x_val is not None:
        dff = dff[dff[xaxis] == x_val]

    if y_val is not None:
        dff = dff[dff[yaxis] == y_val]
    return dff


def create_filter_rows():
    """Create a Dropdown filter for each column.

    These are the fields (columns) that define a unique cohort.
    tablex averages over cohorts if no filters are provided.

    To operate fairly generically, we loop over `MAX_FEATURE_COLS`,
    creating Dropdown filters for existing columns and hidden divs otherwise.

    This routine is only runs on initialization and options are populated through
    a callback. This ensures the data is already loaded.

    """
    cols_per_row = 3
    rows = []
    this_row = []
    for feat_num in range(MAX_FEATURE_COLS):
        this_dropdown = dcc.Dropdown(
            id=f'col{feat_num}-dropdown',
            multi=True,
            style={'display': 'none'}            
        )

        #this_filter = dr.Column(
        #    width=12 // cols_per_row,
        #    children=this_dropdown
        #)
        this_filter = dr.IndicatorColumn(
            width=12 // cols_per_row,
            text='',
            value=this_dropdown,
            id_value=f'col{feat_num}-indicator'
        )
        this_row.append(this_filter)
        if len(this_row) % (cols_per_row) == cols_per_row - 1:
            rows.append(dr.Row(this_row))
            this_row = []

    return html.Div(rows)


def create_graph_row():
    """Create the Graph's after the Filter row has been initialized."""
    children = dr.Row([
        dr.Column(
            width=7,
            children=[
                dr.Row([
                    dr.Column(
                        width=12,
                        children=dcc.RadioItems(
                            id='main-graph-type',
                            options=[
                                {'label': ctype, 'value': ctype} for ctype in ['Heatmap', 'Surface']
                            ],
                            value='Surface',
                            labelStyle={'display': 'inline-block'}
                        )
                    )
                ]),
                dr.Row([
                    dr.Column(
                        width=12,
                        children=dr.dcc.Graph(id='main-graph')
                    )
                ])
            ]
        ),
        dr.Column(
            width=5,
            children=[
                dcc.Graph(id='x-slice'),
                dcc.Graph(id='y-slice'),
                dcc.Graph(id='xy-slice'),
            ],
        )
    ])
    return children


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_filter_rows(),
    create_graph_row()
])


@app.callback(
    [Output('col0-dropdown', 'options'),
     Output('col1-dropdown', 'options'),
     Output('col2-dropdown', 'options'),
     Output('col3-dropdown', 'options'),
     Output('col4-dropdown', 'options'),
     Output('col5-dropdown', 'options'),
     Output('col0-dropdown', 'style'),
     Output('col1-dropdown', 'style'),
     Output('col2-dropdown', 'style'),
     Output('col3-dropdown', 'style'),
     Output('col4-dropdown', 'style'),
     Output('col5-dropdown', 'style'),
     Output('col0-indicator-text', 'children'),
     Output('col1-indicator-text', 'children'),
     Output('col2-indicator-text', 'children'),
     Output('col3-indicator-text', 'children'),
     Output('col4-indicator-text', 'children'),
     Output('col5-indicator-text', 'children')],
    [Input('url', 'href')])
def init_filter_options(href):
    """Initialize the filter options."""
    dff = load_and_filter_data()
    options = []
    styles = []
    children = []
    for feat_num, feat_col in enumerate(feature_cols):
        these_options = [{'value': opt, 'label': opt}
                         for opt in sorted(dff[feat_col].unique())]
        options.append(these_options)
        styles.append(None)
        children.append(f"{feat_col}")

    for empty_opts in range(len(feature_cols), MAX_FEATURE_COLS):
        options.append([])
        styles.append({'display': 'none'})
        children.append(f"")

    return options + styles + children


@app.callback(
    [Output('main-graph', 'figure'),
     Output('main-graph', 'clickData')],
    [Input('col0-dropdown', 'value'),
     Input('col1-dropdown', 'value'),
     Input('col2-dropdown', 'value'),
     Input('col3-dropdown', 'value'),
     Input('col4-dropdown', 'value'),
     Input('col5-dropdown', 'value'),
     Input('main-graph-type', 'value')])
def update_main_on_filter_change(
        col0_filter,
        col1_filter,
        col2_filter,
        col3_filter,
        col4_filter,
        col5_filter,
        graph_type):
    """Update all the plots on any change in filters.
    
    If any features remain, we average over those to produce the
    z(x, y) grid.

    We also render the slices here, but they are the marginal-averages over those dimensions.

    We have separate callbacks to update the slices when the main surface plot is clicked.

    """
    dff = load_and_filter_data(
        col0_filter, col1_filter, col2_filter, col3_filter, col4_filter, col5_filter)

    if dff is None:
        return {'data': []}

    #ipdb.set_trace()

    dff_pivot = dff.set_index(feature_cols + [xaxis])[[yaxis, zaxis]].pivot(columns=yaxis)[zaxis]
    surface = dff_pivot.mean(level=-1) # average over all but the last level (all features)
    if graph_type == 'Surface':
        this_graph = go.Surface(
            z=surface.values,
            x=surface.index,
            y=surface.columns,
            colorscale='viridis',
            opacity=0.85,
            colorbar={'title': zaxis, 'len': .5, 'thickness': 14}
        )
    else:
        this_graph = go.Heatmap(
            z=surface.values,
            x=surface.index,
            y=surface.columns,
            yaxis='y')
    main_graph = {
        'data': [this_graph],
        'layout': go.Layout(
            autosize=True,
            height=MAIN_HEIGHT,
            scene={
                "xaxis": {
                    'title': f"{xaxis}",
                },
                "yaxis": {
                    'title': f"{yaxis}"
                },
                "zaxis": {
                    'title': f'{zaxis}'
                }
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

    reset_clicked = {'points': []}
    
    return main_graph, reset_clicked


@app.callback(
    [Output('x-slice', 'figure'),
     Output('y-slice', 'figure'),
     Output('xy-slice', 'figure')],
    [Input('col0-dropdown', 'value'),
     Input('col1-dropdown', 'value'),
     Input('col2-dropdown', 'value'),
     Input('col3-dropdown', 'value'),
     Input('col4-dropdown', 'value'),
     Input('col5-dropdown', 'value'),
     Input('main-graph', 'clickData')])
def update_slices_on_filter_change(
        col0_filter,
        col1_filter,
        col2_filter,
        col3_filter,
        col4_filter,
        col5_filter,
        clickData):
    """Update all the plots on any change in filters.
    
    If any features remain, we average over those to produce the
    z(x, y) grid.

    We also render the slices here, but they are the marginal-averages over those dimensions.

    We have separate callbacks to update the slices when the main surface plot is clicked.

    """
    try:
        x_val = clickData['points'][0].get('x')
        y_val = clickData['points'][0].get('y')
        xslice_title = f'{yaxis} = {y_val}'
        yslice_title = f'{xaxis} = {x_val}'
        xyslice_title = f'{aa_col} = {x_val + y_val}'
    except(TypeError, IndexError):
        # No point selected: a new feature was added/removed.
        x_val = None
        y_val = None
        xslice_title = f"Average over all '{yaxis}'"
        yslice_title = f"Average over all '{xaxis}'"
        xyslice_title = f"Average over all '{xaxis}', '{yaxis}'"

    dff = load_and_filter_data(
        col0_filter, col1_filter, col2_filter, col3_filter, col4_filter, col5_filter)

    if dff is None:
        return {'data': []}

    #
    # Issue Age (xaxis) slice through fixed Duration (yaxis)
    #
    if x_val is None:
        xslice = dff.groupby(xaxis)[zaxis].mean()
    else:
        xslice = dff[dff[yaxis] == y_val].groupby(xaxis)[zaxis].mean()
    xslice_graph = create_slice(
        x=xslice.index,
        y=xslice.values,
        title=xslice_title,
        layout_kwargs={
            # 'yaxis': {'title': zaxis},
            'xaxis': {'title': xaxis}
        }
    )

    #
    # Duration slice (yaxis) through fixed Issue Age (xaxis)
    #
    if y_val is None:
        yslice = dff.groupby(yaxis)[zaxis].mean()
    else:
        yslice = dff[dff[xaxis] == x_val].groupby(yaxis)[zaxis].mean()
    yslice_graph = create_slice(
        x=yslice.index,
        y=yslice.values,
        title=yslice_title,
        layout_kwargs={
            # 'yaxis': {'title': zaxis},
            'xaxis': {'title': yaxis}
        }
    )

    #
    # Attained age slice through fixed xaxis + yaxis
    #
    if (x_val is not None) and (y_val is not None):
        xy_val = x_val + y_val
        xyslice = dff[dff[aa_col] == xy_val].groupby(xaxis)[zaxis].mean()
        xtitle = f'{xaxis}'
    else:
        xyslice = dff.groupby(aa_col)[zaxis].mean()
        xtitle = f'{aa_col}'
    xyslice_graph = create_slice(
        x=xyslice.index,
        y=xyslice.values,
        title=xyslice_title,
        layout_kwargs={
            'xaxis': {'title': xtitle}
        }
    )

    return xslice_graph, yslice_graph, xyslice_graph


def create_slice(x, y, title, layout_kwargs=None):
    layout = {
        'height': int(MAIN_HEIGHT / 2),
        'margin': {'l': 24, 'b': 30, 'r': 0, 't': 0},
        'annotations': [{
            'x': .1, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
            'xref': 'paper', 'yref': 'paper', 'showarrow': False,
            'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
            'text': title
        }],
        # 'showlegend': True
    }

    if layout is not None:
        layout.update(layout_kwargs)

    return {
        'data': [
            {
                'x': x,
                'y': y,
                'mode': 'lines+markers',
                'name': title
            }
        ],
        'layout': layout
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Table Explorer')
    parser.add_argument('table', type=str, default='qx.csv',
                        help='File containing the tabular data.')

    parser.add_argument('-x', '--xaxis', type=str, default='Issue Age',
                        help='Column (field) name holding the x-axis values.')

    parser.add_argument('-y', '--yaxis', type=str, default='Duration',
                        help='Column (field) name holding the y-axis values.')

    parser.add_argument('-z', '--zaxis', type=str, default='qx',
                        help='Column (field) name holding the z-axis values.')

    args = parser.parse_args()

    df = pd.read_csv(args.table)
    df = df.rename({'Sex': 'Gdr'}, axis=1)
    xaxis = args.xaxis
    yaxis = args.yaxis
    zaxis = args.zaxis
    feature_cols = [col for col in df.columns if col not in [xaxis, yaxis, zaxis]]

    #
    # Look for the Attained Age column, otherwise calculate it
    #
    aa_col = None
    for tryme in AA_ALIASES:
        for col in df.columns:
            if re.match(tryme, col, re.I) is not None:
                aa_col = col
                break
        if aa_col is not None:
            break
    if aa_col is None:
        aa_col = 'Attained Age'
        df[aa_col] = df[xaxis] + df[yaxis]
    else:
        # Attained Age is not a feature, so remove it.
        try:
            feature_cols.remove(aa_col)
        except ValueError:
            pass
        
    app.run_server(debug=True)

