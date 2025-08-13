import dash_bootstrap_components as dbc
import pandas as pd
import dash_ag_grid as dag
from dash import html, dcc, dash_table

def get_users():
    df = pd.read_csv("../data/user_game_playtimes.csv").drop(columns=['Unnamed: 0']).set_index("user_steamid")
    return sorted([str(i) for i in df.index])

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
                                # dbc.Button("About", id="about_app_button", className="header_about_text", color=None),
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
                            dbc.Row(
                                html.Center(
                                    [
                                        html.Div([
                                            html.Div("Steam User ID:", style={'fontWeight': 'bold', 'fontSize': 'large'}),
                                            html.Div(id="user_id_display", style={'fontSize': 'large'}),
                                        ], className="d-flex gap-1")
                                    ],
                                    # id="user_id_display", 
                                    className="d-flex align-items-center justify-content-center"
                                ),
                                style={
                                    'height': '5vh', 
                                }
                            ),
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
                                                            placeholder="Ask about games or ask for recommendations...", 
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
                                    'display': 'block',
                                    'maxHeight': '80vh',
                                }
                            ),
                            dbc.Container(
                                [
                                    dbc.Row([
                                        html.Div(
                                            [
                                                html.H1("Choose User Profile"),
                                                html.I(
                                                    id="user_profile_info_icon",
                                                    style={'marginTop': '10px', 'fontSize': '150%'},
                                                    className="main-menu-icon fa-solid fa-circle-info"
                                                ),
                                                dbc.Tooltip(
                                                    "Use this dropdown to choose a Steam profile to imitate",
                                                    target="user_profile_info_icon",
                                                    placement="right",
                                                ),
                                            ],
                                            style={
                                                'marginTop': '1%'
                                            },
                                            className="d-flex gap-3"
                                        ),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    id="user_profile_dropdown",
                                                    options=[
                                                        {'label': i, 'value': i}
                                                        for i in get_users()
                                                    ],
                                                )
                                            ],
                                            width=4
                                        )
                                    ]),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Br(),
                                                    html.Hr(),
                                                    dbc.Container(
                                                        [
                                                            dbc.Container(
                                                                dag.AgGrid(
                                                                    id="user_profile_table",
                                                                    columnSize="responsiveSizeToFit",
                                                                    className="ag-theme-quartz-dark",
                                                                    dashGridOptions={
                                                                        "rowSelection": {
                                                                            'mode': 'singleRow',
                                                                            'enableClickSelection': True,
                                                                        },
                                                                    },
                                                                    style={'height': '60vh'}
                                                                ),
                                                            ),
                                                            dbc.Container(
                                                                id="user_profile_table_info_area",
                                                                style={'width': '33%', 'display': 'none'}
                                                            ),
                                                        ],
                                                        className="d-flex"
                                                    ),
                                                ],
                                                width=12
                                            ),
                                        ],
                                        id="user_profile_content_row",
                                        style={'display': 'none'}
                                    ),
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