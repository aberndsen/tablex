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
        feature_filters=None,
        x_val=None,
        y_val=None):
    """Filter the global df DataFrame.

    Parameters
    ----------
    feature_filters: dict
       - key = column name
       - value = list
    x_val : int
       Filter the data on `xaxis` == x_val
    y_val : int
       Filter the data on `yaxis` == y_val.

    Returns
    -------
    DataFrame

    """ 
    dff = df
    for feature_name, feature_filter in (feature_filters or {}).items():
        keepers = dff[feature_name].isin(feature_filter)
        dff = dff[keepers]

    if x_val is not None:
        dff = dff[dff[xaxis] == x_val]

    if y_val is not None:
        dff = dff[dff[yaxis] == y_val]
    return dff


def create_filter_rows(suffix='', disabled=False):
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
    for feature_num in range(MAX_FEATURE_COLS):
        this_dropdown = dcc.Dropdown(
            id=f'col{feature_num}-dropdown{suffix}',
            multi=True,
            style={'display': 'none'},
            disabled=disabled
        )

        #this_filter = dr.Column(
        #    width=12 // cols_per_row,
        #    children=this_dropdown
        #)
        this_filter = dr.IndicatorColumn(
            width=12 // cols_per_row,
            text='',
            value=this_dropdown,
            id_value=f'col{feature_num}-indicator{suffix}'
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

def create_e_filter_rows():
    """Elements for the denominator filter row"""
    row = [
        dcc.RadioItems(
            id='do-a-over-e',
            options=[{"label": opt, 'value': opt} for opt in ['No', 'Yes']],
            value='No',
            labelStyle={'display': 'inline-block'})
    ]
    row.extend(create_filter_rows(suffix='-denom', disabled=True).children)

    return html.Div(row)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    html.H3("Cohort filters for Actuals"),
    dcc.Store(id='num-filter-store'),
    create_filter_rows(),

    html.Hr(),
    html.H3("Cohort filters for Denominator (if applicable)"),
    dcc.Store(id='denom-filter-store'),
    dr.Row(
        children=create_e_filter_rows()
    ),

    html.Hr(),
    create_graph_row()
])


@app.callback(
    Output('num-filter-store', 'data'),
    [Input(f'col{num}-dropdown', 'value') for num in range(MAX_FEATURE_COLS)])
def update_num_store(*col_args):
    """Update the filters on the numerator."""
    store = {}
    for feature_num, feature_filter in enumerate(col_args):
        if feature_filter is not None:
            if not isinstance(feature_filter, (list, tuple)):
                feature_filter = [feature_filter]
            store[feature_cols[feature_num]] = feature_filter
    return store


@app.callback(
    Output('denom-filter-store', 'data'),
    [Input(f'col{num}-dropdown-denom', 'value') for num in range(MAX_FEATURE_COLS)])
def update_num_store(*col_args):
    """Update the filters on the denomerator."""
    store = {}
    for feature_num, feature_filter in enumerate(col_args):
        if feature_filter is not None:
            if not isinstance(feature_filter, (list, tuple)):
                feature_filter = [feature_filter]
            store[feature_cols[feature_num]] = feature_filter
    return store


@app.callback(
    [Output('col0-dropdown-denom', 'disabled'),
     Output('col1-dropdown-denom', 'disabled'),
     Output('col2-dropdown-denom', 'disabled'),
     Output('col3-dropdown-denom', 'disabled'),
     Output('col4-dropdown-denom', 'disabled'),
     Output('col5-dropdown-denom', 'disabled')],
    [Input('do-a-over-e', 'value')])
def enable_denom_filters(do_aoe):
    if do_aoe == 'Yes':
        to_ret = [False] * MAX_FEATURE_COLS
    else:
        to_ret = [True] * MAX_FEATURE_COLS
    return to_ret

@app.callback(
    [Output(f'col{num}-dropdown', 'options') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'col{num}-dropdown', 'style') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'col{num}-indicator-text', 'children') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'col{num}-dropdown-denom', 'options') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'col{num}-dropdown-denom', 'style') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'col{num}-indicator-denom-text', 'children') for num in range(MAX_FEATURE_COLS)],
    [Input('url', 'href')])
def init_filter_options(href):
    """Initialize the filter options."""
    dff = load_and_filter_data()
    options = []
    styles = []
    children = []
    for feature_num, feature_name in enumerate(feature_cols):
        these_options = [{'value': opt, 'label': opt}
                         for opt in sorted(dff[feature_name].unique())]
        options.append(these_options)
        styles.append(None)
        children.append(f"{feature_name}")

    for empty_opts in range(len(feature_cols), MAX_FEATURE_COLS):
        options.append([])
        styles.append({'display': 'none'})
        children.append(f"")

    return options + styles + children + options + styles + children


@app.callback(
    [Output('main-graph', 'figure'),
     Output('main-graph', 'clickData')],
    [Input('num-filter-store', 'data'),
     Input('denom-filter-store', 'data'),
     Input('do-a-over-e', 'value'),
     Input('main-graph-type', 'value')])
def update_main_on_filter_change(
        num_filters,
        denom_filters,
        do_aoe,
        graph_type):
    """Update all the plots on any change in filters.
    
    If any features remain, we average over those to produce the
    z(x, y) grid.

    We also render the slices here, but they are the marginal-averages over those dimensions.

    We have separate callbacks to update the slices when the main surface plot is clicked.

    """    
    dff = load_and_filter_data(feature_filters=num_filters)

    if dff is None:
        return {'data': []}

    dff_pivot = dff.set_index(feature_cols + [xaxis])[[yaxis, zaxis]].pivot(columns=yaxis)[zaxis]
    surface = dff_pivot.mean(level=-1) # average over all but the last level (all features)

    if do_aoe == 'Yes':
        dff_denom = load_and_filter_data(feature_filters=denom_filters)
        if len(dff_denom):
            dff_pivot_denom = dff_denom.set_index(
                feature_cols + [xaxis])[[yaxis, zaxis]].pivot(columns=yaxis)[zaxis]
            surface_denom = dff_pivot_denom.mean(level=-1)
            surface = surface.divide(surface_denom)
    
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
            colorscale='viridis')

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
    [Input('num-filter-store', 'data'),
     Input('denom-filter-store', 'data'),
     Input('do-a-over-e', 'value'),
     Input('main-graph', 'clickData')])
def update_slices_on_filter_change(
        num_filters,
        denom_filters,
        do_aoe,
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

    dff = load_and_filter_data(feature_filters=num_filters)

    if dff is None:
        return {'data': []}

    if do_aoe == 'Yes':
        dff_denom = load_and_filter_data(feature_filters=denom_filters)

    #
    # Issue Age (xaxis) slice through fixed Duration (yaxis)
    #
    if y_val is None:
        xslice = dff.groupby(xaxis)[zaxis].mean()
    else:
        if y_val not in dff[yaxis].unique():
            y_val = int(y_val)  # deal with heatmap coordinates
        xslice = dff[dff[yaxis] == y_val].groupby(xaxis)[zaxis].mean()

    if do_aoe == 'Yes':
        if y_val is None:
            xslice_denom = dff_denom.groupby(xaxis)[zaxis].mean()
        else:
            if y_val not in dff[yaxis].unique():
                y_val = int(y_val)  # deal with heatmap coordinates
            xslice_denom = dff_denom[dff_denom[yaxis] == y_val].groupby(xaxis)[zaxis].mean()
        xslice = xslice.divide(xslice_denom)
        
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
    if x_val is None:
        yslice = dff.groupby(yaxis)[zaxis].mean()
    else:
        if x_val not in dff[xaxis].unique():
            x_val = int(x_val)  # deal with heatmap coordinates
        yslice = dff[dff[xaxis] == x_val].groupby(yaxis)[zaxis].mean()

    if do_aoe == 'Yes':
        if x_val is None:
            yslice_denom = dff_denom.groupby(yaxis)[zaxis].mean()
        else:
            if x_val not in dff_denom[xaxis].unique():
                x_val = int(x_val)  # deal with heatmap coordinates
            yslice_denom = dff_denom[dff_denom[xaxis] == x_val].groupby(yaxis)[zaxis].mean()
        yslice = yslice.divide(yslice_denom)

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
    if do_aoe == 'Yes':
        if (x_val is not None) and (y_val is not None):
            xy_val = x_val + y_val
            xyslice_denom = dff_denom[dff_denom[aa_col] == xy_val].groupby(xaxis)[zaxis].mean()
        else:
            xyslice_denom = dff_denom.groupby(aa_col)[zaxis].mean()
        xyslice = xyslice.divide(xyslice_denom)

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

