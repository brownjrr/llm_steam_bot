import os
import dash_bootstrap_components as dbc
from dash import Dash
from layouts import layout
from callbacks import *
import webbrowser


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.LUMEN, 
        dbc.icons.FONT_AWESOME
    ]
)

app.layout = layout
app.title = "Steam Bot"

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        webbrowser.open_new('http://127.0.0.1:8050/')
    app.run(
        debug=True,
        dev_tools_prune_errors=False
    )