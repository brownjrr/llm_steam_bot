import dash_bootstrap_components as dbc
from dash import Dash, html, dcc

def layout():
    return dbc.Container(
        [
            # Triggers
            html.Div(id="ai_loading_message_trigger", style={'display': 'none'}),

            # Store Objects
            dcc.Store(id="last_user_prompt"),

            # Header
            dbc.Navbar(
                dbc.Container(
                    dbc.Row(
                        [
                            dbc.Col(dbc.Label("Steam Bot", className="header_title_text")),
                            dbc.Col(
                                dbc.Button("About", id="about_app_button", className="header_about_text", color=None),
                                width=1,
                                style={
                                    'justify-content': 'center'
                                }
                            ),
                            dbc.Tooltip(
                                "Click here to learn more about this application",
                                target="about_app_button",
                                placement="left",
                            ),
                        ],
                        align="center",
                        justify='between',
                        style={'width': '100%'}
                    ),
                ),
                id="header_navbar",
                color="primary",
                dark=False,
                style={
                    'height': '10%',
                    'padding': 0
                }
            ),
            
            # Body
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Stack(
                            [
                                html.Hr(className='main-menu-hr'),
                                dbc.Button(
                                    html.I(className="main-menu-icon fa-solid fa-house"),
                                    color=None,
                                    id="home-button",
                                ),
                                dbc.Tooltip(
                                    "Home",
                                    target="home-button",
                                    placement="right",
                                ),
                                dbc.Button(
                                    html.I(className="main-menu-icon fa-solid fa-user"),
                                    color=None,
                                    id="profile-button",
                                ),
                                dbc.Tooltip(
                                    "User Profile",
                                    target="profile-button",
                                    placement="right",
                                ),
                            ],
                            gap=4,
                        ),
                        width=1,
                        style={
                            'backgroundColor': 'rgb(206,206,206)'
                        }
                    ),
                    dbc.Col(
                        [
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                id="chat_history_container",
                                                className="chatbox",
                                                width={'offset':1, 'width':2},
                                                style={
                                                    "maxHeight": "100%", 
                                                    "overflow-y": "scroll",
                                                    "display": "flex",
                                                    "flex-direction": "column",
                                                }
                                            ),
                                            dbc.Col(width=1)
                                        ],
                                        id="chat_history_row",
                                        style={
                                            'height': '72vh',
                                            'maxHeight': '72vh',
                                            'marginTop': '5px',
                                        }
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Container(
                                                    [
                                                        dbc.Input(
                                                            id="user-prompt-input",
                                                            placeholder="A large input...", 
                                                            size="lg", 
                                                            style={
                                                                'maxHeight': '80%'
                                                            }
                                                        ),
                                                        dbc.Button(
                                                            "Submit",
                                                            id="submit-prompt-button",
                                                            size='md',
                                                        ),
                                                    ],
                                                    className="d-flex justify-content-center",
                                                )
                                            ),
                                        ],
                                        id="user_input_row",
                                        style={
                                            'height': '20%'
                                        }
                                    ),
                                ],
                                id="home-content",
                                style={
                                    'display': 'block'
                                }
                            ),
                            dbc.Container(
                                [
                                    "Coming Soon!!"
                                ],
                                id="profile-content",
                                style={
                                    'display': 'none',
                                }
                            ),
                        ],
                        width=11,
                    ),
                ],
                id='body_row',
                style={
                    'height': '90vh',
                    'maxHeight': '90vh'
                }
            ),
        ],
        fluid=True,
        style={
            'height': '100vh',
            'width': '100vw',
            'padding': 0,
        }
    )