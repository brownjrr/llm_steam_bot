import requests
import os
import json
import pandas as pd
import numpy as np
import multiprocessing as mp

API_KEY = os.getenv("API_KEY")

def create_raw_apps_table(url):
    # making API request
    req = requests.get(url)
    app_list = req.json()['applist']['apps']

    # creating dataframe
    df = pd.DataFrame([(i['appid'], i['name']) for i in app_list], columns=['appid', 'name'])

    # saving dataframe
    df.to_csv("../../data/raw_game_data.csv", index=False)

    print(df)

def get_game_details(url, app_id_list, chunk_num, verbose=False, save_status_codes=False):
    df = pd.read_csv("../../data/raw_game_data.csv")

    # filter app_id_list
    df = df[df['appid'].isin(app_id_list)]

    print(f'total rows: {df.shape[0]}')
    
    successful_requests = []
    failed_requests = []
    no_data_requests = []
    data_col_set = {'id'}
    data = []
    ret_code_list = []
    for idx in df.index:
        if idx % 1000 == 0:
            print(idx)

        id = df['appid'][idx]

        if verbose: print(f"APP ID: {id}")

        req = requests.get(url+str(id))

        try:
            if verbose: print(req)
            if verbose: print(req.json())

            ret_code_list.append((id, req.json))

            success = req.json()[str(id)]['success']
        except:
            success = None
        if verbose: print(f"Success: {success}")
        
        if success:
            if 'data' in req.json()[str(id)]:
                data_col_set |= set(req.json()[str(id)]['data'].keys())
                data.append({'id': id}|req.json()[str(id)]['data'])
                successful_requests.append(int(id))
            else:
                no_data_requests.append(int(id))
        else:
            failed_requests.append(int(id))

    data_col_set = sorted(list(data_col_set))

    rows = []
    for i in data:
        lst = []
        for col in data_col_set:
            if col in i: lst.append(i[col])
            else: lst.append(None)
        rows.append(lst)
    
    df = pd.DataFrame(rows, columns=data_col_set)
    
    results = {
        'successful_requests': successful_requests,
        'failed_requests': failed_requests,
        'no_data_requests': no_data_requests,
    }

    with open(f"get_game_details_results_{chunk_num}.json", "w+") as f:
        json.dump(results, f)
    
    # reorder columns
    id_cols = ['id', 'steam_appid', 'name']
    df = df[id_cols+[i for i in df.columns if i not in id_cols]]

    print(df)

    if save_status_codes:
        ret_code_df = pd.DataFrame(ret_code_list, columns=['appid', 'request_result'])
        ret_code_df("request_results_failures.csv", index=False)

    # saving data
    df.to_csv(f"../../data/raw_game_details_{chunk_num}.csv", index=False)

def get_appid_list():
    df = pd.read_csv("../../data/raw_game_data.csv")
    return sorted(list(df['appid'].unique()))

def combine_game_details():
    df_list = []
    for i in range(100):
        temp_df = pd.read_csv(f"../../data/raw_game_details_{i}.csv")
        df_list.append(temp_df)
    df = pd.concat(df_list)

    df.to_csv("../../data/raw_game_details.csv", index=False)

def combine_and_analyze_failures(app_detail_url):
    for i in range(100):
        with open(f"get_game_details_results_{i}.json", "r") as f:
            data = json.load(f)
            failed_requests = data['failed_requests'][5:15]

            get_game_details(app_detail_url, failed_requests, 100, verbose=True)
        break

def get_names_of_failures():
    game_df = pd.read_csv("../../data/raw_game_data.csv")

    app_ids = []
    for i in range(100):
        with open(f"get_game_details_results_{i}.json", "r") as f:
            data = json.load(f)
            failed_requests = data['failed_requests']
            app_ids += failed_requests
    
    game_df = game_df[game_df['appid'].isin(app_ids)]

    # save df    
    game_df.to_csv("failed_api_calls.csv", index=False)

def check_status_code_failures(app_detail_url):
    df = pd.read_csv("failed_api_calls.csv")

    get_game_details(app_detail_url, df['appid'].tolist(), 101, verbose=False, save_status_codes=True)


if __name__=="__main__":
    app_list_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    app_detail_url = "http://store.steampowered.com/api/appdetails?appids="
    
    """Step 1"""
    # create_raw_apps_table(app_list_url)
    
    """Step 2"""
    # app_list = get_appid_list()
    # num_chunks = 100
    # chunks = np.array_split(app_list, num_chunks)

    # print(f"Num Chunks: {len(chunks)}")

    # for i, chunk in enumerate(chunks):
    #     get_game_details(app_detail_url, chunk, i)

    """Step 3"""
    combine_game_details()

    """TESTING"""
    # combine_and_analyze_failures(app_detail_url)
    # get_names_of_failures()
    # check_status_code_failures(app_detail_url)
