import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import data
import numpy as np
from dash import ClientsideFunction
from dash import callback_context

# Retrieve options chain and historical price data.
df_aapl = data.get_options_data('AAPL')
df_aapl_hist = data.get_hist_equity_data('AAPL')
trade_date = df_aapl['tradeDate'].iloc[0]

# Build mappings and other globals.
unique_mapping = df_aapl[
    ['dte', 'expirDate']
    ].drop_duplicates().sort_values('dte')
unique_dte = unique_mapping['dte'].tolist()
unique_expirDates = unique_mapping['expirDate'].tolist()
slider_marks = {i: str(dte) for i, dte in enumerate(unique_dte)}
unique_strikes = sorted(df_aapl['strike'].unique())
current_price = df_aapl['stockPrice'].iloc[0]

# Initialize the Dash app.
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Options Selector"),
    html.Div(
        id='slider-tooltip',
        style={'marginBottom': '10px', 'fontSize': '20px'}
    ),
    html.Div(
        dcc.Slider(
            id='dte-slider',
            min=0,
            max=len(unique_dte) - 1,
            value=0,
            step=1,
            marks=slider_marks
        ),
        style={'width': '50%', 'margin': 'auto'}
    ),
    html.Div([
        dcc.RadioItems(
            id='option-type',
            options=[
                {'label': 'Call', 'value': 'Call'},
                {'label': 'Put', 'value': 'Put'}
            ],
            value='Call',
            labelStyle={'display': 'inline-block', 'margin': '10px'}
        )
    ], style={'textAlign': 'center'}),
    dcc.Graph(id='returns-histogram'),
    # Place details and payoff graph side-by-side.
    html.Div([
        html.Div(id='details-div',
                 style={'width': '50%', 'padding': '50px',
                        'padding-left': '10%'}),
        html.Div(
            [
                dcc.RadioItems(
                    id='trade-type',
                    options=[
                        {'label': 'Buy', 'value': 'Buy'},
                        {'label': 'Write', 'value': 'Write'}
                    ],
                    value='Buy',
                    labelStyle={'display': 'inline-block',
                                'margin-right': '10px'}
                ),
                dcc.Graph(id='payoff-graph')
            ],
            style={'width': '50%', 'padding-right': '10%'},
            id='trade-type-container'
        )
    ], style={'display': 'flex', 'marginTop': '10px'}),
    dcc.Store(id='store-bar-click')
], style={'fontFamily': 'Arial, sans-serif'})


# Callback to update the slider tooltip with expiration date.
@app.callback(
    Output('slider-tooltip', 'children'),
    Input('dte-slider', 'value')
)
def update_tooltip(selected_index):
    selected_expirDate = unique_expirDates[selected_index]
    return f"Expiration Date: {selected_expirDate}"


# The update_histogram callback
@app.callback(
    Output('returns-histogram', 'figure'),
    [Input('dte-slider', 'value'),
     Input('option-type', 'value'),
     Input('store-bar-click', 'data')]
)
def update_histogram(selected_index, option_type, clicked_x):
    # Get the selected dte.
    dte = unique_dte[selected_index]

    # Prepare historical data.
    hist_copy = df_aapl_hist.copy()
    hist_copy['future_close'] = hist_copy['Close/Last'].shift(-int(dte))
    hist_copy['pct_return'] = (
        hist_copy['future_close'] / hist_copy['Close/Last']) - 1
    hist_copy = hist_copy[hist_copy['Date'] <= trade_date]

    # Filter the option chain for the selected dte and get strikes directly.
    df_current_opt = df_aapl[df_aapl['dte'] == dte]
    option_strikes = sorted(df_current_opt['strike'].unique())

    hist_ret_min = hist_copy['pct_return'].min()
    hist_ret_max = hist_copy['pct_return'].max()

    # Build a list of (pct_move, strike) pairs
    filtered_pairs = sorted([
        ((strike - current_price) / current_price, strike)
        for strike in option_strikes
        if hist_ret_min <= (strike - current_price)
        / current_price <= hist_ret_max
    ], key=lambda pair: pair[0])

    # if no strikes in the range, use all of them
    if not filtered_pairs:
        filtered_pairs = sorted([
            ((strike - current_price) / current_price, strike)
            for strike in option_strikes
        ], key=lambda pair: pair[0])

    # Unzip the pairs.
    filtered_pct_move = [pair[0] for pair in filtered_pairs]
    filtered_strikes = [pair[1] for pair in filtered_pairs]

    # Use uniform bin edges based on the min and max of the filtered % moves.
    num_bins = len(filtered_pct_move)
    bin_edges = np.linspace(
        min(filtered_pct_move),
        max(filtered_pct_move),
        num_bins + 1
        )

    # Bin the historical % returns using these uniform edges.
    counts, edges = np.histogram(hist_copy['pct_return'], bins=bin_edges)
    bin_midpoints = (np.array(edges[:-1]) + np.array(edges[1:])) / 2

    total_samples = len(hist_copy['pct_return'])
    relative_counts = counts / total_samples
    if option_type == 'Call':
        # For calls, accumulate from the right (higher returns are ITM).
        cdf_values = np.flip(np.cumsum(np.flip(relative_counts)))
    else:
        # For puts, accumulate from the left.
        cdf_values = np.cumsum(relative_counts)

    # Build the option dictionary for bid/ask info.
    if option_type == 'Call':
        option_dict = df_current_opt.set_index('strike')[
            ['callBidPrice', 'callAskPrice']
            ].to_dict('index')
    else:
        option_dict = df_current_opt.set_index('strike')[
            ['putBidPrice', 'putAskPrice']
            ].to_dict('index')

    # For each bin midpoint, find the closest strike's bid/ask info.
    bid_ask_info = []
    for mid in bin_midpoints:
        strike_est = current_price * (1 + mid)
        closest_strike = min(
            option_dict.keys(),
            key=lambda s: abs(s - strike_est)
            )
        info = option_dict[closest_strike]
        if option_type == 'Call':
            bid_ask_info.append(
                f"Bid: ${info['callBidPrice']:.2f}, Ask: ${info['callAskPrice']:.2f}"  # noqa:E501
                )
        else:
            bid_ask_info.append(
                f"Bid: ${info['putBidPrice']:.2f}, Ask: ${info['putAskPrice']:.2f}"  # noqa:E501
                )

    customdata = list(zip(cdf_values, bid_ask_info, filtered_strikes))

    # Use strikes as tick labels
    tickvals = bin_midpoints
    ticktext = [f"${s:.2f}" for s in filtered_strikes]

    # Determine marker colors. Default is blue.
    default_color = "teal"
    marker_colors = []
    for x in tickvals:
        # If clicked_x is defined and matches, mark red.
        if clicked_x is not None and abs(x - clicked_x) < 1e-6:
            marker_colors.append("maroon")
        else:
            marker_colors.append(default_color)

    fig = go.Figure(data=[go.Bar(
        x=tickvals,
        y=counts,
        width=np.diff(edges),
        customdata=customdata,
        marker=dict(color=marker_colors),
        hovertemplate=(
            'Percent Move: %{x:.2%}<br>' +
            'CDF: %{customdata[0]:.2%}<br>' +
            'Bid/Ask: %{customdata[1]}<extra></extra>'
        )
    )])
    fig.update_layout(
        title=f"Distribution of {dte}-Day % Returns ({option_type})",
        xaxis_title="% Return (mapped to strike)",
        yaxis_title="Frequency",
        xaxis=dict(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext
        )
    )
    return fig


# New callback to update details based on which bar is clicked.
@app.callback(
    Output('details-div', 'children'),
    [Input('returns-histogram', 'clickData'),
     Input('option-type', 'value'),
     Input('dte-slider', 'value')]
)
def display_details(clickData, option_type, selected_index):
    ctx = callback_context

    # Default placeholder
    default_message = "Click on a bar for more contract details."

    # No click has occurred
    if clickData is None or not ctx.triggered:
        return default_message

    # Only allow if the callback was triggered by a histogram click
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id != "returns-histogram":
        return default_message

    # Extract the clicked bin's x value (the % move) from clickData
    point = clickData["points"][0]
    strike = point["customdata"][2]

    # Use the selected DTE to filter the option chain.
    dte_val = unique_dte[selected_index]
    df_current_opt = df_aapl[df_aapl['dte'] == dte_val]

    # Find the row whose strike is closest to the estimated strike.
    contract = df_current_opt[df_current_opt['strike'] == strike].iloc[0]

    # Now, based on the option type, extract the corresponding fields.
    if option_type == 'Call':
        bid = contract.get('callBidPrice', 'N/A')
        ask = contract.get('callAskPrice', 'N/A')
        mid_iv = contract.get('callMidIv', 'N/A')
        oi = int(contract.get('callOpenInterest'))
    else:
        bid = contract.get('putBidPrice', 'N/A')
        ask = contract.get('putAskPrice', 'N/A')
        mid_iv = contract.get('putMidIv', 'N/A')
        oi = int(contract.get('putOpenInterest'))

    delta = contract.get('delta', 'N/A')
    gamma = contract.get('gamma', 'N/A')
    theta = contract.get('theta', 'N/A')
    vega = contract.get('vega', 'N/A')

    # Create a two-column table with the contract details.
    details_table = html.Table([
        html.Tr([
            html.Td("Contract Details:"), html.Td("")
            ]),
        html.Tr([
            html.Td("Strike:"), html.Td(f"${contract['strike']:.2f}")
            ]),
        html.Tr([
            html.Td("Option Type:"), html.Td(option_type)
            ]),
        html.Tr([
            html.Td("Bid/Ask:"), html.Td(f"Bid: ${bid:.2f}, Ask: ${ask:.2f}")
            ]),
        html.Tr([
            html.Td("Mid IV:"), html.Td(f"{mid_iv:.2%}")
            ]),
        html.Tr([
            html.Td("Open Interest:"), html.Td(f"{oi:,}")
        ]),
        html.Tr([
            html.Td("Delta:"), html.Td(f"{delta:.2f}")
            ]),
        html.Tr([
            html.Td("Gamma:"), html.Td(f"{gamma:.2f}")
            ]),
        html.Tr([
            html.Td("Theta:"), html.Td(f"{theta:.2f}")
            ]),
        html.Tr([
            html.Td("Vega:"), html.Td(f"{vega:.2f}")
            ]),
    ], style={
        'width': '100%',
        'border': '1px solid #ddd',
        'borderCollapse': 'collapse',
        'textAlign': 'left'
        })

    return details_table


@app.callback(
     Output('trade-type-container', 'style'),
     [Input('returns-histogram', 'clickData'),
      Input('dte-slider', 'value')]
)
def update_payoff_graph_style(clickData, slider_val):
    ctx = callback_context
    if not ctx.triggered:
        # No trigger: hide the graph.
        return {'display': 'none'}
    triggered_id = ctx.triggered[0]['prop_id']
    # If the slider triggered this callback, clear the payoff graph.
    if "dte-slider" in triggered_id:
        return {'display': 'none'}
    # Otherwise, if there is no clickData, also hide it.
    if clickData is None:
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(
    Output('payoff-graph', 'figure'),
    [Input('returns-histogram', 'clickData'),
     Input('option-type', 'value'),
     Input('dte-slider', 'value'),
     Input('trade-type', 'value')]
)
def update_payoff_graph(clickData, option_type, selected_index, trade_type):
    # If no bar is clicked, return an empty figure.
    if clickData is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Payoff Diagram (No Selection)")
        return empty_fig

    # Extract info from the clicked bar.
    point = clickData["points"][0]
    pct_move = point["x"]  # the bin midpoint representing a % move
    strike_est = current_price * (1 + pct_move)

    # Filter the option chain for the selected dte.
    dte_val = unique_dte[selected_index]
    df_current_opt = df_aapl[df_aapl['dte'] == dte_val]

    # Find the contract with strike closest to strike_est.
    idx = (df_current_opt['strike'] - strike_est).abs().idxmin()
    contract = df_current_opt.loc[idx]

    strike = float(contract['strike'])
    if option_type == 'Call':
        bid = float(contract['callBidPrice'])
        ask = float(contract['callAskPrice'])
    else:
        bid = float(contract['putBidPrice'])
        ask = float(contract['putAskPrice'])
    premium = (bid + ask) / 2.0

    # Determine the underlying price range from the option chain.
    option_strikes = sorted(df_current_opt['strike'].unique())
    pct_moves = [((s - current_price) / current_price) for s in option_strikes]
    S_min = current_price * (1 + min(pct_moves))
    S_max = current_price * (1 + max(pct_moves))

    # Create a range for the underlying prices.
    S_range = np.linspace(S_min, S_max, 100)

    # Calculate the payoff based on option type and trade type.
    if option_type == 'Call':
        if trade_type == 'Buy':
            payoff = np.maximum(S_range - strike, 0) - premium
        else:  # Write
            payoff = premium - np.maximum(S_range - strike, 0)
    else:
        if trade_type == 'Buy':
            payoff = np.maximum(strike - S_range, 0) - premium
        else:  # Write
            payoff = premium - np.maximum(strike - S_range, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=payoff, mode='lines', name='Payoff'))
    # Optionally, add a vertical line at the strike.
    fig.add_shape(type='line', x0=strike, x1=strike,
                  y0=min(payoff), y1=max(payoff),
                  line=dict(color='gray', dash='dash'))
    fig.update_layout(
        title=(
            f"Payoff Diagram for Strike ${strike:.2f} (Premium=${premium:.2f}, {trade_type})"  # noqa: E501
            ),
        xaxis_title="Underlying Price at Expiration",
        yaxis_title="Profit / Loss",
        xaxis_range=[S_min, S_max],
        showlegend=False
    )
    return fig


app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='capture_click'),
    Output('store-bar-click', 'data'),
    Input('returns-histogram', 'clickData')
)


if __name__ == '__main__':
    app.run_server(debug=True)
