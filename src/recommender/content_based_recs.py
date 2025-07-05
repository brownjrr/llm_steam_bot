import pandas as pd
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("../data_processing/")

from games import process_game_data

def content_based_recommendation(appid, X, similarity_method=None, top_n=10):
    if similarity_method is None or similarity_method not in ['cosine']:
        assert False, "This function is not capable of handling this similarity method"

    if similarity_method == 'cosine':
        sim_df = pd.DataFrame(cosine_similarity(X), columns=X.index, index=X.index)
    
    # get similarities for appid, sort by similarity score
    app_similarities = sim_df[appid].sort_values(ascending=False)

    # grab top n apps
    app_similarities = app_similarities[:1+top_n].to_frame()

    # change name
    app_similarities.columns = ["score"]

    app_similarities = app_similarities.reset_index()

    # get app names
    game_df = pd.read_csv("../../data/top_100_games.csv")[['appid', 'name']]
    app_similarities = app_similarities.merge(game_df, on=['appid'], how='inner')

    # separate app row from suggestion rows
    app_row = app_similarities[:1]
    app_similarities = app_similarities[1:]

    print(app_row)
    print()
    print(app_similarities)


if __name__ == "__main__":
    game_df = pd.read_csv("../../data/top_100_games.csv")
    game_details_df = pd.read_csv("../../data/top_100_game_details.csv")
    df = process_game_data(game_df, game_details_df, verbose=False)

    content_based_recommendation(70, df, similarity_method='cosine')