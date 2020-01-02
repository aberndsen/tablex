import ipdb
import argparse
import dash
import re

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

import plotly.graph_objects as go

from dash.dependencies import Input, Output, State

from datetime import datetime as dt


#
# Number of columns that are not Issue Age (xaxis), Duration (yaxis) or qx (zaxis)
#
MAX_FEATURE_COLS = 6

MAIN_HEIGHT = 600

#
# Attained Age alias columns, otherwise calculate
# use lower case
AA_ALIASES = ['attained age', 'aa', 'atta', 'atta_age']


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.config.suppress_callback_exceptions = True
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


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("ESU's Table Explorer"),
            #html.H3("Welcome to ESU's cohort explorer Dashboard."),
            html.Div(
                id="intro",
                children=(
                    "Explore your mortality and lapse data by issue age and "
                    "duration using an arbitrary number of descriptor fields "
                    "to segment your population. Click on the surface plot or "
                    "heatmap to visualize behaviour at different time points. "
                    "Finally, look at relative behaviour by enabling the "
                    "denominator (expected) and filtering on a baseline cohort.")
                ,
            ),
        ],
    )

def filter_div(base_id):
    """A standard set of elements for filtering the data.

    This includes {base_id}-title and {base_id}-dropdown elements.
    """

    return html.Div(
        id=f'{base_id}-div',
        children=[
            html.P(
                id=f'{base_id}-title',
                children=''
            ),
            dcc.Dropdown(
                id=f'{base_id}-dropdown',
                multi=True,
                # disabled=True
            )
        ],
        style={'display': 'none'}
    )
            

def generate_filters(prefix=''):
    """

    :return: A Div containing controls for numerator data.
    """
    children = [filter_div(f'{prefix}col{num}') for num in range(MAX_FEATURE_COLS)]
                
    return html.Div(
        id=f"{prefix}control-card",
        children=children
    )


def text_box(header, value, baseid=None):
    """A pretty indicator box with a header and value."""
    if baseid:
        hid = f'{baseid}-header'
        vid = f'{baseid}-value'
    else:
        hid = None
        vid = None
    return html.Div([html.H6(header, id=hid),
                     html.P(value, id=vid)],
                    className='mini_container')


def create_slice_layout(baseid):
    return html.Div([
        html.H6('', id=f'{baseid}-header'),
        html.Div(
            dcc.Loading(
                dcc.Graph(
                    id=f'{baseid}',
                    style={"height": "240px", "width": "100%"},
                ),
            ),
            className='ten columns'
        ),
        html.Div(
            [text_box('Maximum', '', f'{baseid}-max'), text_box('Minimum', '', f'{baseid}-min')],
            className='two columns'
        )],
        className='pretty-container'
    )

        
def create_graph_rows():
    """Create the Graph's after the Filter row has been initialized."""
    children = [
        html.Div([
            text_box(f"Mean", '', 'main-average'),
            text_box(f"Mininum", '', 'main-min'),
            text_box(f"Maximum", '', 'main-max')],
            className='row container-display'
            ),
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='main-graph-type',
                    options=[
                        {'label': ctype, 'value': ctype} for ctype in ['Heatmap', 'Surface']
                    ],
                    value='Surface',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Loading(
                    dcc.Graph(
                        id='main-graph',
                        style={"height": "600px", "width": "100%"},
                    )
                ),
            ],
            className='pretty-container')],
            className='row'),

        #
        html.Hr(),

        #
        # X, Y and XY slices
        #
        create_slice_layout('x-slice'),
        create_slice_layout('y-slice'),
        create_slice_layout('xy-slice'),

    ]
    return html.Div(children)


app.layout = html.Div(
    id='app-container',
    children=[
        dcc.Location(id='url', refresh=False),
        html.Div([
            html.Div(className='two columns'),
            html.Div(
                [description_card()],
                className='eight columns pretty-container'
            )],
            className='row'
        ),

        # Left column
        html.Div(
            id="left-column",
            className="three columns pretty_container",
            children=[
                #description_card(),

                #html.Hr(),
                # Filters for the numerators
                dcc.Store(id='num-filter-store'),
                html.H5('Filters on Actuals'),
                generate_filters(prefix=''),

                html.Hr(),
                # Filters for the denominators
                dcc.Store(id='denom-filter-store'),
                html.H5('Filters on Expected'),
                dcc.Checklist(
                    id='do-a-over-e',
                    options=[
                        {'label': 'Enable', 'value': 'Enable'},
                    ],
                ),
                generate_filters(prefix='denom-')
            ]
        ),

        # Right column
        html.Div(
            id='right-column',
            className='pretty_container nine columns',
            children=[
                create_graph_rows()
            ]
        )
    ]
)


@app.callback(
    [Output(f'col{num}-dropdown', 'options') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'col{num}-div', 'style') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'col{num}-title', 'children') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'denom-col{num}-dropdown', 'options') for num in range(MAX_FEATURE_COLS)]
    + [Output(f'denom-col{num}-title', 'children') for num in range(MAX_FEATURE_COLS)],
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

    return options + styles + children + options + children

@app.callback(
    [Output('main-graph', 'figure'),
     Output('main-average-value', 'children'),
     Output('main-min-value', 'children'),
     Output('main-max-value', 'children'),
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

    if do_aoe == ['Enable']:
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
    return (main_graph,
            surface.mean().mean().round(3),
            surface.min().min().round(3),
            surface.max().max().round(3),
            reset_clicked)



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
    [Input(f'denom-col{num}-dropdown', 'value') for num in range(MAX_FEATURE_COLS)])
def update_denom_store(*col_args):
    """Update the filters on the denomerator."""
    store = {}
    for feature_num, feature_filter in enumerate(col_args):
        if feature_filter is not None:
            if not isinstance(feature_filter, (list, tuple)):
                feature_filter = [feature_filter]
            store[feature_cols[feature_num]] = feature_filter
    return store



@app.callback(
    [Output('x-slice-header', 'children'),
     Output('y-slice-header', 'children'),
     Output('xy-slice-header', 'children')],
    [Input('main-graph', 'clickData')])
def update_slices_on_filter_change(clickData):
    """Update the slice headers."""
    try:
        x_val = clickData['points'][0].get('x')
        y_val = clickData['points'][0].get('y')
        xslice_header = html.B(f'{zaxis}({xaxis} | {yaxis} = {y_val})')
        yslice_header = html.B(f'{zaxis}({yaxis} | {xaxis} = {x_val})')
        xyslice_header = html.B(f'{zaxis}({xaxis} | {aa_col} = {x_val + y_val})')
    except(TypeError, IndexError):
        # No point selected: a new feature was added/removed.
        x_val = None
        y_val = None
        xslice_header = [html.B(f"{zaxis}({xaxis})"), f" averaged over all '{yaxis}'"]
        yslice_header = [html.B(f"{zaxis}({yaxis})"), f" averaged over all '{xaxis}'"]
        xyslice_header = [html.B(f"{zaxis}({xaxis})"), f"  averaged over all '{xaxis}' + '{yaxis}'"]

    return xslice_header, yslice_header, xyslice_header

@app.callback(
    [Output('x-slice', 'figure'),
     Output('x-slice-min-value', 'children'),
     Output('x-slice-max-value', 'children'),
     Output('y-slice', 'figure'),
     Output('y-slice-min-value', 'children'),
     Output('y-slice-max-value', 'children'),
     Output('xy-slice', 'figure'),
     Output('xy-slice-min-value', 'children'),
     Output('xy-slice-max-value', 'children'),
    ],
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
    except(TypeError, IndexError):
        # No point selected: a new feature was added/removed.
        x_val = None
        y_val = None

    dff = load_and_filter_data(feature_filters=num_filters)

    if dff is None:
        return {'data': []}

    if do_aoe == ['Enable']:
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

    if do_aoe == ['Enable']:
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
        title=f'{zaxis}',
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

    if do_aoe == ['Enable']:
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
        title=f'{zaxis}',
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
        xytitle = f'{xaxis}'
    else:
        xyslice = dff.groupby(aa_col)[zaxis].mean()
        xytitle = f'{aa_col}'
    if do_aoe == ['Enable']:
        if (x_val is not None) and (y_val is not None):
            xy_val = x_val + y_val
            xyslice_denom = dff_denom[dff_denom[aa_col] == xy_val].groupby(xaxis)[zaxis].mean()
        else:
            xyslice_denom = dff_denom.groupby(aa_col)[zaxis].mean()
        xyslice = xyslice.divide(xyslice_denom)

    xyslice_graph = create_slice(
        x=xyslice.index,
        y=xyslice.values,
        title=f'{zaxis}',
        layout_kwargs={
            'xaxis': {'title': xytitle}
        }
    )

    return (xslice_graph,
            xslice.min().round(3),
            xslice.max().round(3),
            yslice_graph,
            yslice.min().round(3),
            yslice.max().round(3),
            xyslice_graph,
            xyslice.min().round(3),
            xyslice.max().round(3))


def create_slice(x, y, title, layout_kwargs=None):
    layout = {
        'height': 220, #YYY int(MAIN_HEIGHT / 2),
        'margin': {'l': 24, 'b': 30, 'r': 0, 't': 0},
        #'annotations': [{
        #    'x': .1, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
        #    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
        #    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
        #    'text': title
        #}],
        'showlegend': True
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
            },

            # add a line showing the mean
            {
                'x': x,
                'y': [y.mean().round(3)] * len(y),
                'mode': 'lines',
                'name': 'mean'
            }                
        ],
        'layout': layout
    }


@app.callback(
    [Output(f'denom-col{num}-div', 'style') for num in range(MAX_FEATURE_COLS)],
    [Input('do-a-over-e', 'value')])
def enable_denom_filters(do_aoe):
    """Show the denominator-data filters when enabled."""
    if do_aoe == ['Enable']:
        to_ret = [None] * len(feature_cols)
    else:
        to_ret = [{'display': 'none'}] * len(feature_cols)

    to_ret += [{'display': 'none'}] * (MAX_FEATURE_COLS - len(feature_cols))

    return to_ret



# Run the server
if __name__ == "__main__":
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
    feature_cols = [col for col in df.columns
                    if col not in [xaxis, yaxis, zaxis]]

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

    app.run_server(port=8051, debug=True)
