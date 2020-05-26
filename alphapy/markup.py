import os
import secrets
import logging
import dash

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import peewee as pw

from alphapy.frame import Frame, write_frame
from alphapy.globals import SSEP, PSEP, DATABASE
from dash.dependencies import Input, Output, State
from plotly.graph_objects import layout
from flask import session


#
# Initialize logger
#

logger = logging.getLogger(__name__)


class MarkupTool(object):
    def __init__(self, model, intraday):
        self.intraday = intraday
        self.markup_dir = SSEP.join([model.specs['directory'], 'markup'])
        self.extension = model.specs['extension']
        self.separator = model.specs['separator']

        self.database = pw.SqliteDatabase(DATABASE)
        self.markup = self.get_markup_model()

        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
        self.app.logger = logger
        self.app.server.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(24))
        self.app.server.config["SESSION_FILE_DIR"] = model.specs['directory'] + '/temp'

        self.init_callbacks()
        self.init_layout()
        self.init_db()

    def init_callbacks(self):
        self.app.server.before_request(self.connect_db)
        self.app.server.teardown_request(self.close_db)

        self.app.callback(
            [
                Output("selected-candle-modal", "is_open"),
                Output("selected-candle", "children")
            ],
            [
                Input('candles-graph', 'clickData'),
                Input("close-button", "n_clicks")
            ],
            [
                State("selected-candle-modal", "is_open")
            ]
        )(self.toggle_candle_window)

        self.app.callback(
            Output('candles-graph', 'figure'),
            [
                Input('symbol-dropdown', 'value'),
                Input('candles-graph', 'relayoutData'),
                Input("open-position-checkbox", "checked"),
                Input("buy-button", "n_clicks"),
                Input("sell-button", "n_clicks"),
                Input("delete-button", "n_clicks")
            ]
        )(self.update_candle_figure)

        self.app.callback(
            [
                Output("save-button-alert", "is_open"),
                Output("save-button-alert", "children")
            ],
            [
                Input("save-button", "n_clicks")
            ],
            [
                State("save-button-alert", "is_open")
            ]
        )(self.save_markup)

    def init_layout(self):
        self.app.layout = dbc.Container([
            dbc.Modal(
                [
                    dbc.ModalHeader("Selected candle"),
                    dbc.ModalBody(id='selected-candle'),
                    dbc.ModalFooter(
                        dbc.Row([
                            dbc.Col(
                                dbc.FormGroup(
                                    [
                                        dbc.Checkbox(
                                            id="open-position-checkbox",
                                            className="form-check-input"
                                        ),
                                        dbc.Label(
                                            "Open position?",
                                            html_for="open-position-checkbox",
                                            className="form-check-label"
                                        )
                                    ]
                                )
                            ),
                            dbc.Col([
                                dbc.Button("Buy", color="dark", id="buy-button", n_clicks=0),
                                dbc.Button("Sell", color="danger", id="sell-button", n_clicks=0),
                                dbc.Button("Delete", color="warning", id="delete-button", n_clicks=0),
                                dbc.Button("Close", color="primary", id="close-button", n_clicks=0)
                            ], width="auto")
                        ], justify="between")
                    )
                ], id="selected-candle-modal"
            ),
            dbc.NavbarSimple(
                dbc.Row([
                    dbc.Col(
                        dbc.Select(
                            id="symbol-dropdown",
                            options=[
                                {'label': symbol, 'value': symbol} for symbol in Frame.frames.keys()
                            ],
                            value=next(iter(Frame.frames))
                        ),
                        width="auto"
                    ),
                    dbc.Col(dbc.Button("Save", color="dark", id="save-button", n_clicks=0))
                ]),
                brand="MarketFlow markup tool",
                brand_href="#",
                color="primary",
                dark=True
            ),
            dbc.Alert(
                id="save-button-alert",
                is_open=False,
                duration=3000,
            ),
            dcc.Graph(
                id='candles-graph'
            )
        ])

    def init_db(self):
        self.markup.create_table(fail_silently=True)

    def connect_db(self):
        if self.database.is_closed():
            self.database.connect()

    def close_db(self, exc):
        if not self.database.is_closed():
            self.database.close()

    def get_markup_model(self):
        class Markup(pw.Model):
            symbol = pw.TextField()
            date = pw.TextField()
            high = pw.FloatField()
            low = pw.FloatField()
            buy = pw.BooleanField()
            position = pw.BooleanField()

            class Meta:
                database = self.database
                primary_key = pw.CompositeKey('symbol', 'date')
                table_name = 'markup'

        return Markup

    def upsert_markup(self, markup, symbol, candle, buy, open_position):
        if markup:
            if markup.buy != buy:
                markup.buy = buy
                markup.save()
        else:
            self.markup.insert(symbol=symbol,
                               date=candle['x'],
                               high=candle['high'],
                               low=candle['low'],
                               buy=buy,
                               position=open_position
                               ).execute()

    @staticmethod
    def toggle_candle_window(selected_candle, close_clicks, is_open):
        candle_layout = dbc.ListGroup(horizontal=True)
        if selected_candle:
            selected_candle = selected_candle['points'][0]
            candle_layout.children = [
                dbc.ListGroupItem("Open: {}".format(selected_candle['open'])),
                html.Br(),
                dbc.ListGroupItem("High: {}".format(selected_candle['high'])),
                html.Br(),
                dbc.ListGroupItem("Close: {}".format(selected_candle['close'])),
                html.Br(),
                dbc.ListGroupItem("Low: {}".format(selected_candle['low']))
            ]
        session['selected_candle'] = selected_candle

        return not is_open if selected_candle or close_clicks else is_open, candle_layout

    def update_candle_figure(self, selected_symbol, layout_data, open_position, buy_clicks, sell_clicks, delete_clicks):
        candles = Frame.frames[selected_symbol].df

        session['selected_symbol'] = selected_symbol

        layout_data = layout_data or {}
        autorange = layout_data.get('xaxis.autorange', False)

        xaxis = layout.XAxis(autorange=autorange)
        yaxis = layout.YAxis()

        if not autorange:
            xaxis_range = layout_data.get('xaxis.range', [None, None])
            xaxis_range_0 = layout_data.get('xaxis.range[0]', xaxis_range[0])
            xaxis_range_0 = pd.to_datetime(xaxis_range_0) if xaxis_range_0 else candles.index[-100]
            xaxis_range_1 = layout_data.get('xaxis.range[1]', xaxis_range[1])
            xaxis_range_1 = pd.to_datetime(xaxis_range_1) if xaxis_range_1 else candles.index[-1]
            xaxis.range = [xaxis_range_0, xaxis_range_1]

            candles_slice = candles[xaxis_range_0:xaxis_range_1]
            y_min = candles_slice['low'].min()
            y_max = candles_slice['high'].max()
            yaxis.range = [y_min, y_max]

        if buy_clicks == sell_clicks == delete_clicks == 0:
            session['buy_clicks'] = 0
            session['sell_clicks'] = 0
            session['delete_clicks'] = 0
            session['reset_clicks'] = 0
        else:
            selected_candle = session.get('selected_candle')
            markup = self.markup.get_or_none(symbol=selected_symbol, date=selected_candle['x'])
            open_position = False if open_position is None else open_position

            if sell_clicks > session['sell_clicks']:
                self.upsert_markup(markup, selected_symbol, selected_candle, False, open_position)
                session['sell_clicks'] = sell_clicks
            elif buy_clicks > session['buy_clicks']:
                self.upsert_markup(markup, selected_symbol, selected_candle, True, open_position)
                session['buy_clicks'] = buy_clicks
            elif delete_clicks > session['delete_clicks']:
                if markup:
                    markup.delete_instance()
                session['delete_clicks'] = delete_clicks

        annotations = []
        for markup in (self.markup.select().where(self.markup.symbol == selected_symbol)):
            annotations.append(
                layout.Annotation(
                    xref="x",
                    yref="y",
                    ax=0,
                    ay=20 if markup.buy else -20,
                    x=markup.date,
                    y=markup.low if markup.buy else markup.high,
                    text='B' if markup.buy else 'S',
                    font={
                        'family': 'Courier New, monospace',
                        'size': 12,
                        'color': '#ffffff'
                    },
                    bgcolor='black' if markup.buy else 'red'
                )
            )

        figure_data = go.Candlestick(x=candles.index,
                                     open=candles['open'],
                                     high=candles['high'],
                                     low=candles['low'],
                                     close=candles['close'],
                                     increasing={'line': {'color': 'red'}},
                                     decreasing={'line': {'color': 'black'}})

        figure_layout = go.Layout(
            clickmode='event+select',
            xaxis=xaxis,
            yaxis=yaxis,
            annotations=annotations
        )

        figure = go.Figure(data=[figure_data], layout=figure_layout)

        return figure

    def save_markup(self, save_clicks, is_open):
        if save_clicks:
            selected_symbol = session['selected_symbol']
            query = self.markup.select(self.markup.date,
                                       self.markup.buy,
                                       self.markup.position).where(self.markup.symbol == selected_symbol)
            df = pd.DataFrame(list(query.dicts()))
            if self.intraday:
                df = df.rename(columns={'date': 'datetime'})

            write_frame(df,
                        self.markup_dir,
                        selected_symbol,
                        self.extension,
                        self.separator)

            saved_to = SSEP.join([self.markup_dir, PSEP.join([selected_symbol, self.extension])])
            return not is_open, "Saved to " + saved_to
        return is_open, ''

    def run(self):
        self.app.run_server()
