import dash
import dash_bootstrap_components as dbc
import pandas as pd
import ast
import numpy as np
import requests
from io import BytesIO
from llm import SteamBotModel, get_model
from dash import Input, Output, State, callback, html, dcc, Patch, ALL, ctx
from dash.exceptions import PreventUpdate
from PIL import Image

llm = SteamBotModel(llm=get_model())

# for reading only
user_game_playtimes_df = pd.read_csv("../data/user_game_playtimes.csv").drop(columns=['Unnamed: 0']).set_index("user_steamid")
game_details_df = pd.read_csv("../data/top_1000_game_details.csv")

print(f"user_game_playtimes_df:\n{user_game_playtimes_df}")
print(f"game_details_df:\n{game_details_df}")
print(game_details_df.columns)

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
    State("user_profile_dropdown", "value"),
    prevent_initial_call=True,
)
def populate_ai_response_bubble(_, message_ids, user_prompt, userid):
    id = user_prompt['id']
    response = None
    
    try:
        response = llm.invoke(userid, user_prompt['prompt'])
        pass
    except Exception as e:
        print(f"Error Encountered While Invoking LLM:\nError: {e}")
        
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

@callback(
    Output("user_profile_content_row", "style"),
    Input("user_profile_dropdown", "value"),
    prevent_initial_call=True,
)
def display_user_profile_row(value):
    if value is not None:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@callback(
    Output("user_profile_table", "columnDefs"),
    Output("user_profile_table", "rowData"),
    Input("user_profile_dropdown", "value"),
    prevent_initial_call=True,
)
def populate_user_profile_table(value):
    if value is not None:
        user_row = user_game_playtimes_df.loc[int(value)]
        user_row = user_row[user_row>0].sort_values(ascending=False)
        user_row.name = "Playtime (Minutes)"
        user_row = user_row.to_frame()
        user_row.index.name = "appid"
        game_playtime_df = user_row.reset_index()

        # convert appid to int
        game_playtime_df['appid'] = game_playtime_df['appid'].astype(int)

        # join game details
        game_playtime_df = game_playtime_df.merge(
            game_details_df[['appid', 'name']], 
            on=['appid'],
            how='inner'
        )

        print(f"game_playtime_df:\n{game_playtime_df}")

        # select columns and column order
        game_playtime_df = game_playtime_df[['appid', 'name', 'Playtime (Minutes)']]

        data = game_playtime_df.to_dict('records')
        cols = [{"headerName": i, "field": i} for i in game_playtime_df.columns]

        return cols, data
        
    else:
        raise PreventUpdate

# preprocessing genres column
def process_genres(x):
    if isinstance(x, str):
        genres = [i['description'] for i in ast.literal_eval(x)]
        return ", ".join(genres)
    else:
        return None
        
# processing the developers column
def process_developers(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
        return ", ".join(x)
    else:
        return None
    
# preprocessing publishers column 
def process_publishers(x):
    if isinstance(x, str):
        publishers = ast.literal_eval(x)
        return ", ".join(publishers)
    else:
        return None

# preprocessing metacritic column
def process_metacritic_score(x):
    if isinstance(x, str):
        score_dict = ast.literal_eval(x)
        return float(score_dict['score'])
    else:
        return np.nan

# preprocess release data
def process_release_date(x):
    return pd.to_datetime(
        np.nan if not isinstance(x, str) and (x is None or np.isnan(x)) else ast.literal_eval(x)['date'], 
        errors='coerce'
    ).year

@callback(
    Output("user_profile_table_info_area", "style"),
    Output("user_profile_table_info_area", "children"),
    Input("user_profile_table", "selectedRows"),
    prevent_initial_call=True,
)
def show_hide_info_area(rows):
    patch = Patch()
    if rows:
        row = rows[0]
        patch['display'] = 'flex'

        # show genres, developers, publishers, metacritic,
        # release_date and header_image
        temp_df = game_details_df[game_details_df['appid']==row['appid']]
        items_list = []

        # getting metadata
        genres = temp_df['genres'].values[0]
        genres = process_genres(genres)
        if isinstance(genres, str):
            items_list.append(html.Li(f"Genres: {genres}"))

        developers = temp_df['developers'].values[0]
        developers = process_developers(developers)
        if isinstance(developers, str):
            items_list.append(html.Li(f"Developers: {developers}"))

        publishers = temp_df['publishers'].values[0]
        publishers = process_publishers(publishers)
        if isinstance(publishers, str):
            items_list.append(html.Li(f"Publishers: {publishers}"))

        metacritic = temp_df['metacritic'].values[0]
        metacritic = process_metacritic_score(metacritic)
        if not np.isnan(metacritic):
            items_list.append(html.Li(f"Metatcritic Score: {metacritic}"))

        release_date = temp_df['release_date'].values[0]
        release_date = process_release_date(release_date)
        if not np.isnan(release_date):
            items_list.append(html.Li(f"Release Date: {release_date}"))

        # getting header image
        header_img_path = f"../data/header_images/{row['appid']}.jpg"

        try:
            # #Using Pillow to read the the image
            # pil_img = Image.open(header_img_path)

            print(f"HEADER IMAGE: {temp_df['header_image'].values[0]}")
            response = requests.get(temp_df['header_image'].values[0])

            # Open the image using Pillow from the binary content
            pil_img = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error: {e}")
            pil_img = None

        if items_list:
            if pil_img:
                children = html.Div([
                    html.Img(
                        src=pil_img,
                        className="user_profile_table_img"
                    ),
                    html.Ul(items_list),
                ])
            else:
                children = [
                    html.Ul(items_list),
                ]
        else:
            children = None
    else:
        patch['display'] = 'none'
        children = None

    return patch, children

@callback(
    Output("user_id_display", "children"),
    Input("user_profile_dropdown", "value"),
)
def update_user_id_display(value):
    return f"{str(value)}"
