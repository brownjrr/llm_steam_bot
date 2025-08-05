import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.llm import get_model, summarize_reviews, get_reviews

def eval_llm(original_text, generated_summary):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    emb_orig = model.encode([original_text])
    emb_sum = model.encode([generated_summary])

    similarity = cosine_similarity(emb_orig, emb_sum)[0][0]
    print(f"Semantic similarity score: {similarity}")

    return similarity


if __name__ == '__main__':
    appids = []
    scores = []
    temps = []
    limit = 10
    create_llm_csv = True

    if create_llm_csv:
        all_appids = sorted(list(pd.read_csv('../data/game_player_cnt_ranked_top_1k.csv').sample(limit)['appid'].values))

        for appid in [374320]:
            for temp in [None,None,None,None,None,None,None,None,None,None]:
                for i in range(4):
                    try:
                        original_text = " ".join(x[1] for x in get_reviews(app_id=appid))
                        generated_summary = summarize_reviews(appid=appid, llm=get_model(temp=temp))
                        break
                    except Exception as e:
                        print(appid, e)
                        if i == 3:
                            df = pd.DataFrame({'appid': appids, 'score': scores, 'temp': temps})
                            df.to_csv('../data/llm_responses_single.csv', index=False)
                            raise Exception()

                appids.append(appid)
                scores.append(eval_llm(original_text, generated_summary))
                temps.append(temp)

        df = pd.DataFrame({'appid': appids, 'score': scores, 'temp': temps})
        df.to_csv('../data/llm_responses_single.csv', index=False)

    df = pd.read_csv('../data/llm_responses.csv')
    df = df.sort_values(by='score', ascending=False)
    df['appid'] = df['appid'].astype(str)
    print(df)
