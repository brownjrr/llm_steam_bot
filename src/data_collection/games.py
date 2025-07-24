import requests
import json
import pandas as pd
import numpy as np
import glob
import time
import cv2


def create_raw_apps_table(url):
    # making API request
    req = requests.get(url)
    app_list = req.json()['applist']['apps']

    # creating dataframe
    df = pd.DataFrame([(i['appid'], i['name']) for i in app_list], columns=['appid', 'name'])

    # saving dataframe
    df.to_csv("../../data/raw_game_data.csv", index=False)

    print(df)

def get_game_details(url, app_id_list, chunk_num=None, verbose=False, save_status_codes=False):
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
    for i, idx in enumerate(df.index):
        if i % 1000 == 0:
            print(i)

        id = df['appid'][idx]

        if verbose: print(f"APP ID: {id}")

        req = requests.get(url+str(id))

        try:
            if verbose: print(req)
            if verbose: print(req.json())
            
            with open("appid_request_log.txt", "a+") as f:
                f.write(f"App ID: {id}, Status Code: {req}\n")

            ret_code_list.append((id, req.json))

            success = req.json()[str(id)]['success']
        except Exception as e:
            print(f"Error processing app ID {id}: {e}")
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

    if chunk_num is not None:
        json_filename = f"get_game_details_results_{chunk_num}.json"
    else:
        json_filename = "get_game_details_results.json"

    with open(json_filename, "w+") as f:
        json.dump(results, f)
    
    # reorder columns
    id_cols = ['id', 'steam_appid', 'name']
    df = df[id_cols+[i for i in df.columns if i not in id_cols]]

    print(df)

    if save_status_codes:
        ret_code_df = pd.DataFrame(ret_code_list, columns=['appid', 'request_result'])
        ret_code_df("request_results_failures.csv", index=False)
    
    # saving data
    if chunk_num:
        df.to_csv(f"../../data/raw_game_details_{chunk_num}.csv", index=False)
    else:
        df.to_csv(f"../../data/raw_game_details.csv", index=False)

def get_game_details_v2(url, app_id_list, chunk_num=None, verbose=False, save_status_codes=False):
    df = pd.read_csv("../../data/raw_game_data.csv")

    # check for successful requests
    successful_requests = [int(i.split('\\')[-1].replace(".json", "")) for i in glob.glob("../../data/successful_requests/*")]

    # check for failed requests
    failed_requests = [int(i.split('\\')[-1].replace(".json", "")) for i in glob.glob("../../data/failed_requests/*")]

    # check for no data requests
    no_data_requests = [int(i.split('\\')[-1].replace(".json", "")) for i in glob.glob("../../data/no_data_requests/*")]

    print(f"successful_requests: {successful_requests}")
    print(f"failed_requests: {failed_requests}")
    print(f"no_data_requests: {no_data_requests}")

    # filter app_id_list
    df = df[(df['appid'].isin(app_id_list)) & (~df['appid'].isin(successful_requests+failed_requests+no_data_requests))]

    print(f'total rows: {df.shape[0]}')

    # define the number of requests to make per minute
    interval = 60 / 40 # we want 40 requests every minute (200 every 5 minutes)
    # interval = 1 # we want 40 requests every minute (400 every 5 minutes)

    # iterate through records
    for i, idx in enumerate(df.index):
        if i % 1000 == 0:
            print(i)

        # get appid
        id = df['appid'][idx]

        if verbose: print(f"APP ID: {id}")

        # make request
        req = requests.get(url+str(id))

        # check if request was successful
        try:
            success = req.json()[str(id)]['success']
        except Exception as e:
            print(f"Error: {e}")
            print(f"URL: {url+str(id)}")
            print(req)
            print(req.json())
            assert False

        
        # if request was successful, save data to json file
        if success is True and 'data' in req.json()[str(id)]:
            data = req.json()[str(id)]['data']

            with open(f"../../data/successful_requests/{id}.json", "w+") as f:
                json.dump(data, f)
        elif success is True and 'data' not in req.json()[str(id)]:
            data = req.json()
            data[str(id)]['status_code'] = req.status_code

            with open(f"../../data/no_data_requests/{id}.json", "w+") as f:
                json.dump(data, f)
        elif success is False:
            data = req.json()
            data[str(id)]['status_code'] = req.status_code

            with open(f"../../data/failed_requests/{id}.json", "w+") as f:
                json.dump(data, f)

        time.sleep(interval)


def get_appid_list():
    df = pd.read_csv("../../data/raw_game_data.csv")
    return sorted(list(df['appid'].unique()))

def combine_game_details():
    df_list = []
    for file in glob.glob("../../data/raw_game_details*.csv"):
        print(file)
        df_list.append(pd.read_csv(file))
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

def get_image(df, appid, verbose=False):
    df = df[df['appid']==appid]
    img_url = df['header_image'].values[0]

    # get image from url
    response = requests.get(img_url)
    img_arr = np.asarray(bytearray(response.content))

    # decode image
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # resize image
    image = cv2.resize(image, (800, 600))

    if verbose:
        # Display the image (optional)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image
    
def get_game_header_images(df):
    app_ids = list(df['appid'].unique())

    for appid in app_ids:
        img = get_image(df, appid, verbose=False)
        
        # Save the image to a new file
        cv2.imwrite(f'../../data/header_images/{appid}.jpg', img)

if __name__=="__main__":
    app_list_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    app_detail_url = "http://store.steampowered.com/api/appdetails?appids="
    
    """Processing image data"""
    df = pd.read_csv("../../data/top_100_game_details.csv")

    get_game_header_images(df)

    # """Step 1"""
    # # create_raw_apps_table(app_list_url)
    
    # """Step 2"""
    # app_list = get_appid_list()
    
    # # app_list = app_list[:50000]
    # app_list = app_list[50000:150000]
    # # app_list = app_list[150000:200000]
    # # app_list = app_list[200000:250000]
    # # app_list = app_list[250000:]

    # num_chunks = 10
    # chunks = np.array_split(app_list, num_chunks)

    # print(f"Num Chunks: {len(chunks)}")

    # for i, chunk in enumerate(chunks):
    #     # get_game_details(app_detail_url, chunk, i)
    #     get_game_details_v2(app_detail_url, chunk, i)

    # """Step 3"""
    # # combine_game_details()

    """Top 100 Games Data"""
    # get app ids
    # df = pd.read_csv("../../data/top_100_games.csv")
    # app_list = df['appid'].tolist()
    # get_game_details_v2(app_detail_url, app_list)

    
    """
    These games are missing from AppList API call. Loading them manually
    """
    # app_list = [1089350, 227940, 755790, 901583]

    # # define the number of requests to make per minute
    # interval = 60 / 40 # we want 40 requests every minute (200 every 5 minutes)

    # for id in app_list:
    #     # make request
    #     req = requests.get(app_detail_url+str(id))

    #     # check if request was successful
    #     try:
    #         success = req.json()[str(id)]['success']
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         print(f"URL: {app_detail_url+str(id)}")
    #         print(req)
    #         print(req.json())
    #         assert False

        
    #     # if request was successful, save data to json file
    #     if success is True and 'data' in req.json()[str(id)]:
    #         data = req.json()[str(id)]['data']

    #         with open(f"../../data/successful_requests/{id}.json", "w+") as f:
    #             json.dump(data, f)
    #     elif success is True and 'data' not in req.json()[str(id)]:
    #         data = req.json()
    #         data[str(id)]['status_code'] = req.status_code

    #         with open(f"../../data/no_data_requests/{id}.json", "w+") as f:
    #             json.dump(data, f)
    #     elif success is False:
    #         data = req.json()
    #         data[str(id)]['status_code'] = req.status_code

    #         with open(f"../../data/failed_requests/{id}.json", "w+") as f:
    #             json.dump(data, f)

    #     time.sleep(interval)

    """TESTING"""
    # df = pd.read_csv("../../data/raw_game_data.csv")
    # print(df)
    # df = pd.read_csv("../../data/raw_game_details.csv")
    # print(df.shape)
    # combine_and_analyze_failures(app_detail_url)
    # get_names_of_failures()
    # check_status_code_failures(app_detail_url)
