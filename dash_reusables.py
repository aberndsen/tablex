"""A collection of reusable DASH-components.

This is inspired by the 'Reusable Components' in the
DASH getting-started guide:
https://dash.plot.ly/getting-started

But, together with /static/gfast-sitecustom2.css gives some
MFC/JH branding

"""
import colorlover
import dash_core_components as dcc
import dash_html_components as html

from textwrap import dedent
COLORSCALE = []
COLORSCALE2 = {}
_topick = ['Reds', 'Greens', 'Blues', 'Greys', 'Purples', 'Oranges']
for n, clr in enumerate(_topick):
    dtn = -1 * ((n + 1) % 2)
    this = colorlover.scales['9']['seq'][_topick[n]][2:]
    if dtn:
        this = this[::dtn]
    COLORSCALE.extend(this)
    COLORSCALE2[n] = this
COLORSCALE = dict((n, color) for n, color in enumerate(COLORSCALE))
NCOLORS = 7

DIVISION_COLORS = {
    'Asia': 'rgba(255, 152, 0, .9)',
    'Canadian': 'rgba(158, 158, 158, .8)',
    'Group': 'rgba(13, 105, 66, .75)',
    'US': 'rgba(18, 72, 119, .75)',
    'unknown': 'rgba(205, 22, 22, .6)'
}


def get_mfc_colors(i, N, division=False, fallback_color='grey'):
    """Get a color.

    Arguments
    ---------
    i : int
        index (sequence number) for this item.
    N : int
        Total number of items for this group/sequence.
    division : bool
        Use the divisional colors, or a nice colorscale.
    fallback_color : str
        Color to use if we don't find a match.

    Returns
    -------
    rgba : str

    """
    if division is True:
        this_color = DIVISION_COLORS.get(i, fallback_color)
    else:
        NSPAN = 14  # 21 = # of reds + greens + blues, 14 = # reds + greens
        if N < NSPAN:
            # span the reds, greens and blues
            idx = (i * NSPAN) // N
        else:
            idx = (i * len(COLORSCALE)) // n

        this_color = COLORSCALE.get(idx, fallback_color)
    return this_color


#    _ _ _                      _     _
#   |  __ \                    | |   | |
#   | |__) |___ _   _ ___  __ _| |__ | | _ _  _ _
#   |  _  // _ \ | | / __|/ _` | '_ \| |/ _ \/ __|
#   | | \ \  __/ |_| \__ \ (_| | |_) | |  __/\__ \
#   |_|  \_\___|\__,_|___/\__,_|_.__/|_|\___||___/
#


def AccordionPanel(children, title, id, closed=False):
    """https://www.w3schools.com/howto/howto_js_accordion.asp.

    An AccordionPanel is a
    Row, with a panel header (title) and collapsible content (children).

    Arguments
    ---------
    children : list
       list of html elements to put in the panel
    title : str or list
       items to place in the header of the panel
    id : str
       name of the panel
    closed : bool
       start the panel closed

    Notes
    -----
    For each "AccordionPanel" `id`, you must create two callbacks
    @app.callback(
    Output(id+'_panel', 'style'),
    [Input(id, 'n_clicks')])
    def id_panelshowhide(n_clicks):
        if n_clicks % 2 == 1:
            return {'display': 'none'}
        else:
            return {'display': 'block'}

    and

    @app.callback(
    Output(id, 'className'),
    [Input(id, 'n_clicks')])
    def id_plusminus(n_clicks):
        if n_clicks % 2 == 1:
            return "plus"
        else:
            return "minus"

    Be sure to add a callback for this id.

    """
    panel_id = id + '_panel'
    if closed:
        this_style = {'display': 'none'}
    else:
        this_style = {'display': 'block'}
    if not isinstance(children, (list, tuple)):
        children = [children]

    return Row(
        Column(
            html.Div(
                className='accordion',
                children=[
                    html.Button(
                        children=title,
                        id=id,
                        n_clicks=0,
                    ),
                    html.Div(
                        children=children,
                        className="panel",
                        id=panel_id,
                        style=this_style
                    )
                ]
            ),
            width=12
        )
    )


def Column(children=None, width=1, **kwargs):
    """Create a CSS-styled Column of `width` (/12)."""
    number_mapping = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
        12: 'twelve'
    }
    return html.Section(
        children,
        className="{} columns".format(number_mapping[width]),
        **kwargs
    )


def Details(children, summary=None):
    """Details component, to render collapsible tips."""
    return html.Details([
        html.Summary(summary),
        html.Div(
            children=children,
            className="detail_info"
        )
    ])


# returns top indicator div
def IndicatorColumn(width, text, value, id_value):
    if isinstance(value, (list, tuple)):
        this_value = value
    else:
        this_value = [value]
    return Column(
        width=width,
        children=html.Div([
            html.Div(
                className="indicator",
                children=[
                    html.P(
                        children=text,
                        id=f"{id_value}-text",
                        className="twelve columns indicator_text"
                    ),
                    html.Div(
                        id=id_value,
                        className="indicator_value",
                        children=this_value
                    )
                ]
            )],
            style={'padding': '6px 6px 0px 6px'}
        )
    )


def Row(children=None, **kwargs):
    """Create a CSS-styled Row/Div."""
    return html.Div(
        children,
        className="row",
        **kwargs
    )


def create_menu(menu_items=None):
    """Inspired by https://codepen.io/erikterwan/pen/EVzeRP

    Arguments
    ---------
    menu_items : list
       list of records. If the record is a dict, we expect
       'href' and 'label' keys.
       otherwise we use the same label as the record.

    """
    item_content = []
    if menu_items is not None:
        for item in menu_items:
            if isinstance(item, dict):
                href = item.get('href', '')
                label = item.get('label', '')
            else:
                href = item
                label = item
            this = html.A(
                children=html.Li(label),
                href=href,
                target='_blank'
            )
            item_content.append(this)
        item_content = html.Ul(
            id='menu',
            children=item_content
        )
    this = html.Nav(
        role="navigation",
        children=[
            html.Div(
                id="menuToggle",
                # A fake / hidden checkbox is used as click reciever,
                # so you can use the :checked selector on it.
                children=[
                    dcc.Input(type="checkbox"),
                    html.Span(),
                    html.Span(),
                    html.Span(),
                    item_content
                ]
            )
        ]
    )

    return this


def gfast_footer():
    # footer-ish
    return html.Div(
        children=[
            Row([
                Column(
                    # html.A(html.Img(src="/static/GFAST_final_logo.png"), href='mailto:gfast@manulife.com'),
                    width=12,
                    style={
                        'background-color': 'rgb(82,82,82)',
                        'border-radius': '25px',
                        # 'background-image': 'url(/assets/mfc_scarf.png)',
                        'background-size': 'cover',
                        'background-position': 'center'
                    },
                    children=[
                        html.Div(
                            style={'text-align': 'center'},
                            children=[
                                html.Img(
                                    src="/assets/GFAST_final_logo.png",
                                    style={
                                        # 'background': '#ededed',
                                        'width': '15%',
                                        'vertical-align': 'middle'
                                    }
                                )
                            ]
                        )
                    ]
                )
            ],
                style={'padding': '25px'}
            ),
            Row([
                Column(
                    width=12,
                    children=[html.Small(
                        dcc.Markdown(
                            dedent("""
The information on this dashboard is designed to provide helpful information on the usage and costs
associated with modelling and computing on the Actuarial Compute Environment.

For exact cost and billing information, refer to the
[ACE Consumption Reports](https://mfc.sharepoint.com/sites/GblFin/AMC/Reference/Pages/ACE%20Consumption%20Reports/ACE%20Consumption%20Reports.aspx).""")  # noqa: E501
                        )
                    )],
                    style={
                        'text-align': 'center',
                        'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto',
                        'padding': '3px 30px 3px 30px'
                    }
                ),
            ])
        ]
    )
