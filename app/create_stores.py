from llm import get_reviews, get_review_retriever, get_game_data_retriever
import pandas as pd
from tqdm import tqdm


"""Run this code to populate vectore stores. THis should only be run ONCE"""
if __name__ == "__main__":
    pass
    get_game_data_retriever()
    for appid in tqdm(pd.read_csv('../data/game_player_cnt_ranked_top_1k.csv')['appid'].values):
        get_review_retriever(get_reviews(appid), skip_populating=False, filter_app_id=appid)
