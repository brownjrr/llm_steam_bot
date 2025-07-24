from llm import get_reviews, get_review_retriever
import pandas as pd
from tqdm import tqdm


"""Run this code to populate vectore stores. THis should only be run ONCE"""
if __name__ == "__main__":
    #pass
    for appid in tqdm(pd.read_csv('../data/game_player_cnt_ranked_top_1k.csv')['appid'].values):
        get_review_retriever(get_reviews(appid), skip_populating=False, filter_app_id=appid)
