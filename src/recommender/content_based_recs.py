import pandas as pd
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("../data_processing/")

from games import process_game_data


def content_based_recommendation(appids, X, sim_df=None, similarity_method=None, top_n=10):
    if similarity_method is not None and similarity_method not in ['cosine']:
        assert False, "This function is not capable of handling this similarity method"

    if sim_df is not None:
        if similarity_method == 'cosine':
            sim_df = pd.DataFrame(cosine_similarity(X), columns=X.index, index=X.index)
    
    # removing any unfound app ids from appids list
    errant_ids = []
    for _id in appids:
        if _id not in sim_df.columns:
            errant_ids.append(_id)
    
    if len(errant_ids) > 0: 
        # print(f"Could not find the following ids: {errant_ids}")
        for _id in errant_ids:
            appids.remove(_id)

    # get similarities for appid, sort by similarity score
    app_similarities = sim_df[appids]

    # average app_similarities columns
    mean_app_similarities = app_similarities.mean(axis=1).sort_values(ascending=False)

    # # grab top n apps
    # app_similarities = mean_app_similarities[:len(appids)+top_n].to_frame()
    app_similarities = mean_app_similarities.to_frame()

    # change name of similarity score column
    app_similarities.columns = ["score"]
    app_similarities = app_similarities.reset_index()

    # get app names
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")[['appid', 'name']]
    app_similarities = app_similarities.merge(game_df, on=['appid'], how='left')

    # separate app row from suggestion rows
    app_rows = app_similarities[app_similarities['appid'].isin(appids)]
    app_similarities = app_similarities[~app_similarities['appid'].isin(appids)]
    if top_n is not None: app_similarities = app_similarities[:top_n]

    return app_similarities

if __name__ == "__main__":
    game_df = pd.read_csv("../../data/top_100_games.csv")
    game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    img_summary_df = pd.read_csv("../../data/top_1000_game_image_summary.csv")

    df = process_game_data(
        game_df, 
        game_details_df, 
        img_summary_df=img_summary_df, 
        verbose=False, 
        include_image_summary=True
    )

    content_based_recommendation(
        [70, 291550], 
        df, 
        similarity_method='cosine'
    )