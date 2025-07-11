import dash
import dash_bootstrap_components as dbc
import time
import sys
import os
from llm import SteamBotModel, get_reviews, get_model
from dash import Input, Output, State, callback, html, dcc, Patch, ALL, ctx
from dash.exceptions import PreventUpdate


llm = SteamBotModel(llm=get_model())

def generate_user_message(text):
    return dbc.Card(
        dcc.Markdown(text),
        style={
            "width": "max-content",
            "font-size": "14px",
            "padding": "0px 0px",
            "border-radius": 15,
            "margin-bottom": 5,
            "margin-left": "auto",
            "margin-right": 0,
            "max-width": "80%",
        },
        body=True,
        color='primary',
        inverse=True
    )

def generate_ai_loading_message(prompt_id):
    return dbc.Card(
        html.Div(dbc.Spinner(size="sm")),
        id={'type': 'ai_prompt_response', 'index': prompt_id},
        style={
            "width": "max-content",
            "font-size": "14px",
            "padding": "0px 0px",
            "border-radius": 15,
            "margin-bottom": 5,
            "margin-left": 0,
            "margin-right": "auto",
                "max-width": "80%",
        },
        body=True,
        color="#F5F5F5",
        inverse=False,
    )

@callback(
    Output("chat_history_container", "children", allow_duplicate=True),
    Output("last_user_prompt", "data"),
    Output("user-prompt-input", "value"),
    Input("submit-prompt-button", "n_clicks"),
    Input("user-prompt-input", "n_submit"),
    State("user-prompt-input", "value"),
    prevent_initial_call=True,
)
def add_user_response(n, n_sub, prompt):
    if (n_sub or n) and prompt is not None and prompt != "":
        # calculating id of user_prompt
        if n_sub is None: n_sub = 0
        if n is None: n = 0
        id = n + n_sub

        patched = Patch()
        patched.append(generate_user_message(prompt))
        return patched, {'prompt': prompt, 'id': id}, None
    
    raise PreventUpdate

@callback(
    Output("chat_history_container", "children", allow_duplicate=True),
    Output("ai_loading_message_trigger", "children"),
    Input("last_user_prompt", "data"),
    prevent_initial_call=True,
)
def add_ai_response_loading(user_prompt):
    if user_prompt is not None:
        patched = Patch()
        patched.append(generate_ai_loading_message(user_prompt['id']))
        return patched, None

    raise PreventUpdate

@callback(
    Output({'type': 'ai_prompt_response', 'index': ALL}, "children"),
    Input("ai_loading_message_trigger", "children"),
    State({'type': 'ai_prompt_response', 'index': ALL}, "id"),
    State("last_user_prompt", "data"),
    prevent_initial_call=True,
)
def populate_ai_response_bubble(_, message_ids, user_prompt):
    id = user_prompt['id']

    response = None
    try:
        response = llm.invoke(user_prompt['prompt'])
    except Exception as e:
        print(e)
        
        response = """
## ⚠️ Error Encountered

Something didn't go as expected.  
Try another input
"""

    # print(f"response:\n{response}")

    return [
        dcc.Markdown(response) if i['index']==id else dash.no_update 
        for i in message_ids
    ]

@callback(
    Output("home-content", "style"),
    Output("profile-content", "style"),
    Input("home-button", "n_clicks"),
    Input("profile-button", "n_clicks"),
    prevent_initial_call=True,
)
def display_page_content(n1, n2):
    if not ctx.triggered_id:
        triggered_id = 'No clicks yet'
    else:
        triggered_id = ctx.triggered_id
    
    patched_home = Patch()
    patched_profile = Patch()

    if triggered_id == "home-button":
        patched_home['display'] = 'block'
        patched_profile['display'] = 'none'
    elif triggered_id == "profile-button":
        patched_home['display'] = 'none'
        patched_profile['display'] = 'block'
    else:
        raise PreventUpdate

    return patched_home, patched_profile
