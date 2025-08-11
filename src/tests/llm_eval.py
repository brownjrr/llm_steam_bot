import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.llm import get_model, summarize_reviews, get_reviews

def eval_llm(original_text, generated_summary):
    precision, recall, F1 = score([generated_summary], [original_text], model_type='distilbert-base-uncased')

    model = SentenceTransformer('all-MiniLM-L6-v2')

    emb_orig = model.encode([original_text])
    emb_sum = model.encode([generated_summary])

    cosine = cosine_similarity(emb_orig, emb_sum)[0][0]
    print(f"Semantic similarity score: {cosine}")

    return precision.mean().item(), recall.mean().item(), F1.mean().item(), cosine


if __name__ == '__main__':
    appids = []
    precision_scores = []
    recall_scores = []
    F1_scores = []
    cosine_scores = []
    temps = []
    limit = 10

    all_appids = sorted(list(pd.read_csv('../data/game_player_cnt_ranked_top_1k.csv').sample(limit)['appid'].values))
    
    for appid in [374320, 413150, 294100, 671860, 47890, 582160, 252490, 391540, 214950, 292730]:
        original_text = " ".join(x[1] for x in get_reviews(app_id=appid))
        for i in range(4):
            try:
                generated_summary = summarize_reviews(appid=appid, llm=get_model())
                break
            except Exception as e:
                print(appid, e)

        appids.append(appid)
        precision, recall, F1, cosine = eval_llm(original_text, generated_summary)
        precision_scores.append(precision)
        recall_scores.append(recall)
        F1_scores.append(F1)
        cosine_scores.append(cosine)

    df = pd.DataFrame({'appid': appids, 'precision': precision_scores, 'recall': recall_scores, 'F1': F1_scores, 'cosine': cosine_scores})
    df.to_csv('../data/llm_responses_all_metrics.csv', index=False)
