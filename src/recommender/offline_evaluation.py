import pandas as pd
import numpy as np
import json
import sys
import glob
import ast
import warnings
import multiprocessing
from itertools import product
from uuid import uuid4
from sklearn.metrics import ndcg_score
from content_based_recs import content_based_recommendation
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

sys.path.append("../data_processing/")

from games import process_game_data

warnings.filterwarnings("ignore")

def get_train_test(verbose=False):
    with open("../../data/offline_evaluation/train.json", "r") as f:
        train_set = json.load(f)
    
    with open("../../data/offline_evaluation/test.json", "r") as f:
        test_set = json.load(f)
    
    # remove users from train and test set
    # (removing those without a sample in train or test)
    del_keys  = []
    for key in train_set:
        user = key
        num_games_train = len(train_set[key])
        num_games_test = len(test_set[key])

        if num_games_train==0 or num_games_test==0: # ignore these users
            if verbose: print(f"Deleting user ({user}):\n\t{num_games_train} games in the train set\n\t{num_games_test} games in the test set")
            del_keys.append(key)
    
    for key in del_keys:
        del train_set[key]
        del test_set[key]

    return train_set, test_set

def recommendation_hit(user, test, rec_df):
    # get appids from the test set
    try:
        test_appids = sorted(
            [(i, test[user][i]) for i in test[user]], 
            key=lambda x: x[1],
            reverse=True
        )
    except:
        # print(test)
        # print(list(test.keys()))
        print([i for i in test.keys() if i.startswith("76561199382275")])
        print(str(user) in list(test.keys()))
        print(int(user) in list(test.keys()))
        assert  False

    test_appids = [int(i[0]) for i in test_appids]
    hits = set(rec_df['appid']).intersection(set(test_appids))

    return len(hits)>0

def get_precision_recall_at_k(user, test, rec_df, n):
    """
    Find Precision@k and Recall@k
    """

    recommended_appids = rec_df['appid'].tolist()
    relevant_appids = [int(i[0]) for i in sorted(list(test[user].items()), reverse=True, key=lambda x: x[1])]
    doc_relevant = [1 if i in relevant_appids else 0 for i in recommended_appids]

    if n is None or n > len(recommended_appids):
        true_n = len(recommended_appids)
    else:
        true_n = n
    
    # print(f"len(recommended_appids): {len(recommended_appids)}")
    # print(f"n: {n}")
    # print(f"true_n: {true_n}")

    pre_at_n = sum(doc_relevant[:true_n]) / len(doc_relevant[:true_n])
    rec_at_n = sum(doc_relevant[:true_n]) / len(relevant_appids)

    return pre_at_n, rec_at_n

def get_ndcg(user, train, test, rec_df, k=10):
    """
    Find Normalized Discounted Cumulative Gain
    """
    owned_games = [int(i) for i in train[user]]
    owned_games_test = [int(i) for i in test[user]]
    full_appid_lst = rec_df['appid'].sort_values().tolist()
    full_appid_lst = [i for i in full_appid_lst if i not in owned_games]

    if len(full_appid_lst)>1:
        true_relevance = np.asarray([[1 if i in owned_games_test else 0 for i in full_appid_lst]])
        pred_relevance = np.asarray([[
            rec_df[rec_df['appid']==i]['score'].values[0] if i in rec_df['appid'].tolist() else 0 
            for i in full_appid_lst
        ]])

        # print(f"true_relevance: {true_relevance}")
        # print(f"pred_relevance: {pred_relevance}")

        ndcg_at_k = ndcg_score(true_relevance, pred_relevance, k=k)

        # print(f"ndcg_at_k: {ndcg_at_k}")
    else:
        ndcg_at_k = 0

    return ndcg_at_k

def get_mrr(user, train, test, rec_df, k=10):
    ranked_list = rec_df['appid'].tolist()
    # print(f"ranked_list:\n{ranked_list}")

    relevant_items = [int(i[0]) for i in sorted(list(test[user].items()), reverse=True, key=lambda x: x[1])]
    # print(f"relevant_items:\n{relevant_items}")

    found_relevant = False
    for j, item in enumerate(ranked_list):
        if item in relevant_items:
            reciprocal_rank = (1 / (j + 1))  # j+1 for 1-based indexing
            found_relevant = True
            break
    if not found_relevant:
        reciprocal_rank = 0  # No relevant item found

    return reciprocal_rank

def evaluate_content_based_recommendations(
        processed_game_data, train, test, similarity_method, top_n_train_examples=None,
        top_n_recommendations=None, return_hit_rate=True, return_mean_avg_prec=True, return_ndcg=True,
        return_full=False
    ):
    users = list(train.keys())
    data = []

    if similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )

    data_cols = ['userid']

    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')
    
    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")

        # get appids from the train set
        train_appids = sorted(
            [(i, train[user][i]) for i in train[user]], 
            key=lambda x: x[1],
            reverse=True
        )

        train_appids = [int(i[0]) for i in train_appids]

        if top_n_train_examples is not None:
            train_appids = train_appids[:top_n_train_examples]

        
        full_rec_df = content_based_recommendation(
            train_appids, # appids
            processed_game_data, # processed_game_data
            sim_df=sim_df,
            top_n=None
        )

        # print(f"full_rec_df:\n{full_rec_df}")

        # top top_n_recommendations
        rec_df = full_rec_df.head(top_n_recommendations)
        
        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    if return_full:
        return results_df
    
    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)
    
    return tuple(return_list)

def get_model():
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        # model="us.meta.llama4-maverick-17b-instruct-v1:0",
        model="us.meta.llama3-1-70b-instruct-v1:0",
        region_name="us-east-1"
    )

    return llm

def get_llm_recommendations(train, test, df, unseen_only=False):
    train_data = {user: list(train[user].items()) for user in train}

    if unseen_only:
        complete_recommendations = {i.split("\\")[-1].split(".")[0] for i in glob.glob("llm_recommendations/*.json")}
        unseen = set(train_data.keys()).difference(complete_recommendations)
        train_data = {i: train_data[i] for i in train_data if i in unseen}
    
    app_dict = df[['appid', 'name']].set_index('appid').to_dict()['name']
    # print(app_dict)
    
    def get_app_name(id):
        return app_dict[id]
    
    llm = get_model()

    # create embedding
    embeddings = HuggingFaceEmbeddings()

    # defining vector store
    vector_store = Chroma(
        collection_name="temp_game_names",
        embedding_function=embeddings,
    )

    # Create documents from df records
    loader = CSVLoader(
        file_path="../../data/game_player_cnt_ranked_top_1k.csv", 
        encoding='utf-8',
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['name']
        }
    )

    documents = loader.load()

    print(f"Num documents: {len(documents)}")

    # adding documents to vector store
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    print("Grabbing Retriever")
    
    # transforming chroma vectorsotre into a retriever
    retriever = vector_store.as_retriever()

    for idx, user in enumerate(train_data):
        if idx % 1 == 0:
            print(f"{idx} of {len(train_data)}")
        
        recommendation_dict = dict()
        temp_list = train_data[user]
        new_list = []
        ids = []
        for i in temp_list:
            _id = int(i[0])
            ids.append(_id)
            new_list.append((get_app_name(_id), i[1]))
        train_data[user] = new_list

        # Defining prompt template
        template = """
        You need to give your top 20 NEW game recommendations for 
        a player who has played the following games (each item in the list has the 
        name of a game and the amount of time played, in minutes):

        Games Played: {games_played}

        Games In Corpus: {context}

        Return recommendations as a Python list of tuples where
        the first element in the tuple is game's appid and the second element
        is a score showing how much you recommend this game. Make sure each appid in this
        list is unique. If you can't recommend 20 games, recommend as many as you can.
        Return this list with no other words. Sort this list from your strongest to 
        weakest recommendation.
        """

        # Creating prompt
        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm
        try:
            response = chain.invoke({'context': retriever, 'games_played': train_data[user]})
        except Exception as e:
            print(f"Failed to get recommendations for User: {user}")
            print(e)
            assert False
            continue

        recommendation_dict[user] = response.content

        with open(f"llm_recommendations/{user}.json", "w+") as f:
            json.dump(recommendation_dict, f)

def content_based_offline_eval(return_full=False):
    # get data tables
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    img_summary_df = pd.read_csv("../../data/top_1000_game_image_summary.csv")
    img_summary_df_v2 = pd.read_csv("../../data/top_1000_game_image_keywords.csv")
    screenshot_summary_df = pd.read_csv("../../data/top_1000_game_screenshot_summary.csv")

    train, test = get_train_test()

    # # process game metadata and unstructured data
    # df = process_game_data( # not including llm generated feature
    #     game_df, 
    #     game_details_df, 
    #     verbose=False, 
    #     include_image_summary=False
    # ) # Hit Rate: 0.6014022331861854 | Mean Avg Precision: 0.13019994806543755

    # df_w_img_summary = process_game_data( # tone of header image
    #     game_df, 
    #     game_details_df, 
    #     img_summary_df=img_summary_df, 
    #     verbose=False, 
    #     include_image_summary=True
    # ) # Hit Rate: 0.5738769150869903 | Mean Avg Precision: 0.12516229550766034

    # df_w_img_keywords = process_game_data( # header image keywords
    #     game_df, 
    #     game_details_df, 
    #     img_summary_df=None, 
    #     img_summary_df_v2=img_summary_df_v2, 
    #     verbose=False, 
    #     include_image_summary=False
    # ) # Hit Rate: 0.6276291872240977 | Mean Avg Precision: 0.14172942092962867

    # df_w_screenshot_keywords = process_game_data( # screenshot keywords
    #     game_df, 
    #     game_details_df, 
    #     img_summary_df=None, 
    #     img_summary_df_v2=None, 
    #     screenshot_summary_df=screenshot_summary_df,
    #     verbose=False, 
    #     include_image_summary=False
    # ) # Hit Rate: 0.6372370812775903 | Mean Avg Precision: 0.14427421448974292

    header_screenshot_df = process_game_data( # screenshot keywords
        game_df, 
        game_details_df, 
        img_summary_df=None, 
        img_summary_df_v2=img_summary_df_v2, 
        screenshot_summary_df=screenshot_summary_df,
        verbose=False, 
        include_image_summary=False
    ) # -> Hit Rate: 0.654375486886523 | Mean Avg Precision: 0.1555959491041288

    df_list = [
        # (df, "Normal"), 
        # (df_w_img_summary, "Image Summary (Tone)"), 
        # (df_w_img_keywords, "Image Summary (Keywords)"), 
        # (df_w_screenshot_keywords, "Screenshot Summary (Keywords)"),
        (header_screenshot_df, "Screenshot and Header Image Keywords"),
        # (None, "LLM"),
    ]

    if not return_full:
        data = []
        for i, name in df_list:
            print(f"Name: {name}")

            for n_recs in [1, 5, 10, 20]:
                print(f"Number of Reccommendations: {n_recs}")
                hit_rate, mean_avg_prec, ndcg = evaluate_content_based_recommendations(
                    i, train, test, similarity_method="cosine", top_n_train_examples=5,
                    top_n_recommendations=n_recs, return_hit_rate=True, return_mean_avg_prec=True, 
                    return_ndcg=True, return_full=return_full
                )

                print(f"================================================================")

                data.append((name, n_recs, hit_rate, mean_avg_prec, ndcg))

        df = pd.DataFrame(
            data, 
            columns=['name', 'k', 'hit_rate', 'mean_avg_prec', 'ndcg']
        )
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"offline_eval_results_{now_str}.csv", index=False)
    else:
        return_df_list = []
        for i, name in df_list:
            print(f"Name: {name}")

            for n_recs in [1, 5, 10, 20]:
                print(f"Number of Reccommendations: {n_recs}")
                results_df = evaluate_content_based_recommendations(
                    i, train, test, similarity_method="cosine", top_n_train_examples=5,
                    top_n_recommendations=n_recs, return_hit_rate=True, return_mean_avg_prec=True, 
                    return_ndcg=True, return_full=return_full
                )
                
                results_df['k'] = n_recs
                return_df_list.append(results_df)

                print(f"================================================================")
        
        df = pd.concat(return_df_list)
        df.to_csv("content_based_offline_eval_results_full.csv", index=False)


def collect_recommendations_llm(unseen_only=False):
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    train, test = get_train_test()
    get_llm_recommendations(train, test, game_df, unseen_only=unseen_only)

def get_game_appid(df, name):
    temp_df = df[df['name']==name]
    if not temp_df.empty:
        appid = temp_df['appid'].values[0]
    else:
        appid = None
    return appid

def llm_recommendation_offline_eval(top_n_recommendations, return_hit_rate=True, return_mean_avg_prec=True, return_ndcg=True):
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    train, test = get_train_test()

    data = []
    data_cols = ['userid']

    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')

    for user, games in test.items():
        with open(f"llm_recommendations/{user}.json", "r") as f:
            recommendations = json.load(f)

        recommendation_str = recommendations[user].strip()
        if recommendation_str[-2] != ")": recommendation_str = recommendation_str.replace("]", ")]")
        if "assistant<|end_header_id|>" in recommendation_str: recommendation_str = recommendation_str.replace("assistant<|end_header_id|>", "")
        if recommendation_str[-1] != "]": recommendation_str += "]"
        
        recommendations[user] = ast.literal_eval(recommendation_str)
        recommendations[user] = sorted(
            [i for i in recommendations[user] if isinstance(i, tuple) and i[0] is not None], 
            key=lambda x: x[1], 
            reverse=True
        )

        # ensure recommendations[user] has only two columns
        recommendations[user] = [(i[0], i[1]) for i in recommendations[user]]

        # print(f"recommendations[{user}]: {recommendations[user]}")

        # create recommendation df
        rec_df = pd.DataFrame(recommendations[user], columns=['appid', 'score'])

        # top top_n_recommendations
        rec_df = rec_df.head(top_n_recommendations)
        
        # print(f"rec_df:\n{rec_df}\n=================")

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)
    
    return tuple(return_list)

def llm_offline_eval():
    data = []
    for n_recs in [1, 5, 10, 20]:
        print(f"Number of Reccommendations: {n_recs}")
        
        hit_rate, map, ndcg = llm_recommendation_offline_eval(n_recs)

        print(f"================================================================")

        data.append(("Llama-3.1 Recommendations", n_recs, hit_rate, map, ndcg))

    df = pd.DataFrame(
        data, 
        columns=['name', 'k', 'hit_rate', 'mean_avg_prec', 'ndcg']
    )

    print(df)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"llm_offline_eval_results_{now_str}.csv", index=False)

def get_collaborative_filtering_data(train):
    data = [(user, game, train[user][game]) for user in train for game in train[user]]

    interaction_df = pd.DataFrame(data, columns=["user_steamid", "appid", "playtime_forever"])
    interaction_df['appid'] = interaction_df['appid'].astype(int)
    interaction_df = interaction_df.sort_values(by=["user_steamid", "appid"])

    # Binarize playtime as implicit feedback (1 if played > x minutes)
    interaction_df["interaction"] = (interaction_df["playtime_forever"] > 1200).astype(int)

    # Pivot to a user-item matrix
    user_item_matrix = interaction_df.pivot_table(
        index="user_steamid", columns="appid", values="interaction", fill_value=0
    )

    # Compute item-item similarity using cosine similarity
    item_user_matrix = user_item_matrix.T  # Transpose to get items x users
    item_similarity = cosine_similarity(item_user_matrix)

    # Build item similarity DataFrame for exploration
    item_similarity_df = pd.DataFrame(
        item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index
    )

    return interaction_df, user_item_matrix, item_similarity_df

def recommend_games_for_user(user_id, interaction_df, user_item_matrix, item_similarity_df, game_details_df, top_n=5):
    # Get the user’s interaction vector
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found.")
        return []

    #user_vector = user_item_matrix.loc[user_id]
    #interacted_games = user_vector[user_vector > 0].index.tolist() # Only recommended? So more than X hours? Better to filter all played?
    # Instead of using binarized matrix, use original interaction_df:
    played_games = interaction_df[interaction_df["user_steamid"] == user_id]["appid"].unique()

    # Store recommendation scores
    scores = pd.Series(dtype=float)

    for game in played_games:
        similar_scores = item_similarity_df[game]
        scores = scores.add(similar_scores, fill_value=0)

    # Remove games the user has already interacted with
    #scores = scores.drop(labels=interacted_games, errors="ignore")
    # Remove all games the user ever played, not just liked ones
    scores = scores.drop(labels=played_games, errors="ignore")


    # Get top N recommendations
    top_recommendations = scores.sort_values(ascending=False).head(top_n)
    top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
    #print(top_recommendations)
    top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
    #print(top_recommendations)
    top_recommendations = top_recommendations[['appid','name','similarity_score']]

    print(top_recommendations)
    return top_recommendations

def collaborative_offline_eval(
    test, interaction_df, item_similarity_df, game_details_df, top_n=5,
    return_hit_rate=True, return_mean_avg_prec=True, return_ndcg=True,
    return_full=False, return_mrr=True, 
):
    
    data = []
    data_cols = ['userid']
    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')
    if return_mrr: data_cols.append('MRR')

    for idx, user_id in enumerate(test):
        if idx % 500 == 0:
            print(f"{idx} of {len(test)}")

        # Instead of using binarized matrix, use original interaction_df:
        played_games = interaction_df[interaction_df["user_steamid"] == user_id]["appid"].unique()

        # Store recommendation scores
        scores = pd.Series(dtype=float)

        for game in played_games:
            similar_scores = item_similarity_df[game]
            scores = scores.add(similar_scores, fill_value=0)

        # Remove games the user has already interacted with
        #scores = scores.drop(labels=interacted_games, errors="ignore")
        # Remove all games the user ever played, not just liked ones
        scores = scores.drop(labels=played_games, errors="ignore")

        # Get top N recommendations
        top_recommendations = scores.sort_values(ascending=False).head(top_n)
        top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
        #print(top_recommendations)
        top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
        #print(top_recommendations)
        rec_df = top_recommendations[['appid','name','similarity_score']]
        rec_df = rec_df.rename(columns={'similarity_score': 'score'})
        
        data_lst = [user_id]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user_id, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user_id, test, rec_df, top_n)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user_id, train, test, rec_df, k=top_n)
            data_lst.append(ndcg)
        if return_mrr:
            rr = get_mrr(user_id, train, test, rec_df, k=top_n)
            data_lst.append(rr)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    if return_full:
        return results_df
    
    print(f"results_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    if return_mrr:
        mean_recip_rank = np.mean(results_df['MRR'].tolist())
        print(f"Mean Reciprocal Rank: {mean_recip_rank}")
        return_list.append(mean_recip_rank)
        

    return tuple(return_list)

def collaborative_offline_eval_multiple_k(
    test, interaction_df, item_similarity_df, game_details_df,
    return_hit_rate=True, return_mean_avg_prec=True, return_ndcg=True,
    return_full=False, return_mrr=True
):
    if not return_full:
        data = []
        for n_recs in [1, 5, 10, 20]:
            print(f"Number of Reccommendations: {n_recs}")
            
            hit_rate, map, ndcg, mrr = collaborative_offline_eval(
                test, 
                interaction_df,  
                item_similarity_df, 
                game_details_df, 
                top_n=n_recs,
                return_hit_rate=return_hit_rate, 
                return_mean_avg_prec=return_mean_avg_prec, 
                return_ndcg=return_ndcg,
                return_full=return_full,
                return_mrr=return_mrr,
            )

            print(f"================================================================")

            data.append(("Memory-Based Collaborative Filtering", n_recs, hit_rate, map, ndcg, mrr))

        df = pd.DataFrame(
            data, 
            columns=['name', 'k', 'hit_rate', 'mean_avg_prec', 'ndcg', 'mrr']
        )

        print(df)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"collaborative_offline_eval_results_{now_str}.csv", index=False)
    else:
        df_list = []
        for n_recs in [1, 5, 10, 20]:
            print(f"Number of Reccommendations: {n_recs}")
            
            results_df = collaborative_offline_eval(
                test, 
                interaction_df,  
                item_similarity_df, 
                game_details_df, 
                top_n=n_recs,
                return_hit_rate=return_hit_rate, 
                return_mean_avg_prec=return_mean_avg_prec, 
                return_ndcg=return_ndcg,
                return_full=return_full,
                return_mrr=return_mrr,
            )

            results_df['k'] = n_recs
            df_list.append(results_df)

            print(f"================================================================")
        df = pd.concat(df_list)

        print(f"results_df:\n{df}")

        assert False

        df.to_csv("collaborative_offline_eval_results_full.csv", index=False)
    

def hybrid_model(
    processed_game_data, train, test, interaction_df, 
    item_similarity_df, game_details_df, similarity_method, weight, top_n_train_examples=None, 
    top_n_recommendations=None, return_hit_rate=True, return_mean_avg_prec=True, 
    return_ndcg=True
):
    users = list(train.keys())
    data = []
    data_cols = ['userid']
    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')

    if similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )

    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")

        scaler = MinMaxScaler()

        """Content Based Recommendations"""
        # get appids from the train set
        train_appids = sorted(
            [(i, train[user][i]) for i in train[user]], 
            key=lambda x: x[1],
            reverse=True
        )

        train_appids = [int(i[0]) for i in train_appids]

        if top_n_train_examples is not None:
            train_appids = train_appids[:top_n_train_examples]

        
        full_rec_df = content_based_recommendation(
            train_appids, # appids
            processed_game_data, # processed_game_data
            sim_df=sim_df,
            top_n=None
        )

        # print(f"full_rec_df:\n{full_rec_df}")

        # top top_n_recommendations
        content_rec_df = full_rec_df.head(top_n_recommendations)
        content_rec_df[['score']] = scaler.fit_transform(content_rec_df[['score']])
        content_appids = set(content_rec_df['appid'])

        # print(f"content_rec_df:\n{content_rec_df}")

        """Collaborative Recommendations"""
        # Instead of using binarized matrix, use original interaction_df:
        played_games = interaction_df[interaction_df["user_steamid"] == user]["appid"].unique()

        # Store recommendation scores
        scores = pd.Series(dtype=float)

        for game in played_games:
            similar_scores = item_similarity_df[game]
            scores = scores.add(similar_scores, fill_value=0)

        # Remove all games the user ever played, not just liked ones
        scores = scores.drop(labels=played_games, errors="ignore")

        # Get top N recommendations
        top_recommendations = scores.sort_values(ascending=False).head(top_n_recommendations)
        top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
        top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
        collab_rec_df = top_recommendations[['appid','name','similarity_score']]
        collab_rec_df = collab_rec_df.rename(columns={'similarity_score': 'score'})
        collab_rec_df[['score']] = scaler.fit_transform(collab_rec_df[['score']])
        collab_appids = set(collab_rec_df['appid'])

        # print(f"collab_rec_df:\n{collab_rec_df}")

        """Combining Models"""
        content_rec_df['source'] = 'content-based'
        collab_rec_df['source'] = 'collab'
        combined_df = content_rec_df.merge(collab_rec_df, on="appid", how="outer", suffixes=('_content', '_collab'))
        combined_df[['name_content', 'name_collab']] = combined_df[['name_content', 'name_collab']].replace({np.nan: None})
        combined_df[['score_collab', 'score_content']] = combined_df[['score_collab', 'score_content']].fillna(0)

        def compute_final_score(row):
            return row['score_collab']*weight + row['score_content']*(1-weight)
        combined_df['combined_score'] = combined_df.apply(compute_final_score, axis=1)
        
        def get_name(row):
            if row['name_content'] is None:
                return row['name_collab']
            elif row['name_collab'] is None:
                return row['name_content']
            else:
                return row['name_content']
            
        combined_df['name'] = combined_df.apply(get_name, axis=1)
        rec_df = combined_df[['appid', 'name', 'combined_score']]
        rec_df = rec_df.rename(columns={'combined_score': 'score'})

        """Getting Results"""

        # print(f"rec_df:\n{rec_df}")

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    return tuple(return_list)

def hybrid_model_v2(
    processed_game_data, train, test, interaction_df, 
    item_similarity_df, game_details_df, similarity_method, top_n_train_examples=None, 
    top_n_recommendations=None, return_hit_rate=True, return_mean_avg_prec=True, 
    return_ndcg=True, game_threshold=5
):
    users = list(train.keys())
    data = []
    data_cols = ['userid']
    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')

    if similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )

    # get interactions per game
    popularity_df = interaction_df['appid'].value_counts()

    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")
        
        playtimes = list(train[user].values())
        num_relevant_games = sum([1 if i > 1200 else 0 for i in playtimes])
        num_irrelevant_games = len(playtimes) - num_relevant_games
        irrelevant_pct = num_irrelevant_games / len(playtimes)
        
        # use_collab = True if irrelevant_pct < game_threshold[0] and len(playtimes) > game_threshold[1] else False
        use_collab = True if len(playtimes) > game_threshold else False
        # use_collab = True if num_relevant_games >= game_threshold else False
        # use_collab = True if irrelevant_pct <= game_threshold else False

        # print(f"User '{user}' has {num_relevant_games} relevant games")

        if use_collab:
            """Collaborative Recommendations"""
            # Instead of using binarized matrix, use original interaction_df:
            played_games = interaction_df[interaction_df["user_steamid"] == user]["appid"].unique()

            # Store recommendation scores
            scores = pd.Series(dtype=float)

            for game in played_games:
                similar_scores = item_similarity_df[game]
                scores = scores.add(similar_scores, fill_value=0)

            # Remove all games the user ever played, not just liked ones
            scores = scores.drop(labels=played_games, errors="ignore")

            # Get top N recommendations
            top_recommendations = scores.sort_values(ascending=False).head(top_n_recommendations)
            top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
            top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
            
            rec_df = top_recommendations[['appid','name','similarity_score']]
            rec_df = rec_df.rename(columns={'similarity_score': 'score'})

            # print(f"collab_rec_df:\n{rec_df}")
        else:
            """Content Based Recommendations"""
            # get appids from the train set
            train_appids = sorted(
                [(i, train[user][i]) for i in train[user]], 
                key=lambda x: x[1],
                reverse=True
            )

            train_appids = [int(i[0]) for i in train_appids]

            if top_n_train_examples is not None:
                train_appids = train_appids[:top_n_train_examples]

            
            full_rec_df = content_based_recommendation(
                train_appids, # appids
                processed_game_data, # processed_game_data
                sim_df=sim_df,
                top_n=None
            )

            # top top_n_recommendations
            rec_df = full_rec_df.head(top_n_recommendations)

            # print(f"content_rec_df:\n{rec_df}")

        """Getting Results"""

        # print(f"rec_df:\n{rec_df}")

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    return tuple(return_list)

def hybrid_model_v3(
    processed_game_data, train, test, interaction_df, 
    item_similarity_df, game_details_df, similarity_method, top_n_train_examples=None, 
    top_n_recommendations=None, return_hit_rate=True, return_mean_avg_prec=True, 
    return_ndcg=True, pop_threshold=5
):
    users = list(train.keys())
    data = []
    data_cols = ['userid']
    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')

    if similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )

    print(f"sim_df:\n{sim_df}")

    # get interactions per game
    popularity_counts = interaction_df['appid'].value_counts()
    popular_items = popularity_counts[popularity_counts >= pop_threshold].index
    niche = popularity_counts[popularity_counts < pop_threshold].index

    print(f"popular_items:\n{popular_items}")
    print(f"niche:\n{niche}")
    
    scaler = MinMaxScaler()

    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")
        
        """Collaborative Recommendations"""
        # Instead of using binarized matrix, use original interaction_df:
        temp_interaction_df = interaction_df[interaction_df["appid"].isin(list(popular_items))]
        played_games = temp_interaction_df[temp_interaction_df["user_steamid"] == user]["appid"].unique()

        # print(f"Collaborative: {len(played_games), played_games}")
    
        # Store recommendation scores
        scores = pd.Series(dtype=float)

        for game in played_games:
            similar_scores = item_similarity_df[game]
            scores = scores.add(similar_scores, fill_value=0)

        # Remove all games the user ever played, not just liked ones
        scores = scores.drop(labels=played_games, errors="ignore")

        # Get top N recommendations
        top_recommendations = scores.sort_values(ascending=False).head(top_n_recommendations)
        top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
        top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
        collab_rec_df = top_recommendations[['appid','name','similarity_score']]
        collab_rec_df = collab_rec_df.rename(columns={'similarity_score': 'score'})
        collab_rec_df[['score']] = scaler.fit_transform(collab_rec_df[['score']])

        # print(f"collab_rec_df:\n{collab_rec_df}")

        """Content Based Recommendations"""
        # get appids from the train set
        train_appids = sorted(
            [(i, train[user][i]) for i in train[user] if i in list(niche)], 
            key=lambda x: x[1],
            reverse=True
        )

        train_appids = [int(i[0]) for i in train_appids]

        if len(train_appids) > 0:
            print(f"Content Based: {len(train_appids), train_appids}")
            assert False

        if top_n_train_examples is not None:
            train_appids = train_appids[:top_n_train_examples]

        
        full_rec_df = content_based_recommendation(
            train_appids, # appids
            processed_game_data, # processed_game_data
            sim_df=sim_df,
            top_n=None
        )

        # top top_n_recommendations
        content_rec_df = full_rec_df.head(top_n_recommendations)

        # print(f"content_rec_df:\n{content_rec_df}")

        # combine recommendations
        rec_df = pd.concat([collab_rec_df, content_rec_df]).drop_duplicates(subset=['appid'])

        # print(f"rec_df: {rec_df}")

        """Getting Results"""

        # print(f"rec_df:\n{rec_df}")

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    return tuple(return_list)

def switch_hybrid_model(
    processed_game_data, train, test, interaction_df, 
    item_similarity_df, game_details_df, similarity_method='cosine', sim_df=None, 
    top_n_train_examples=None, top_n_recommendations=None, return_hit_rate=True, 
    return_mean_avg_prec=True, return_ndcg=True, return_mrr=True, return_full=False, 
    game_threshold=5, num_content_recs=200
):
    users = list(train.keys())
    data = []
    data_cols = ['userid']

    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')
    if return_mrr: data_cols.append('MRR')

    if sim_df is None and similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )

    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")
        
        played_games = interaction_df[interaction_df["user_steamid"] == user]["appid"].unique()

        if len(played_games) <= game_threshold:
            """Content Based Recommendations"""
            # get appids from the train set
            train_appids = sorted(
                [(i, train[user][i]) for i in train[user]], 
                key=lambda x: x[1],
                reverse=True
            )

            train_appids = [int(i[0]) for i in train_appids]

            if top_n_train_examples is not None:
                train_appids = train_appids[:top_n_train_examples]

            full_rec_df = content_based_recommendation(
                train_appids, # appids
                processed_game_data, # processed_game_data
                sim_df=sim_df,
                top_n=None
            )

            # top top_n_recommendations
            content_rec_df = full_rec_df.head(num_content_recs)

            # print(f"content_rec_df:\n{content_rec_df.sort_values(by='appid')}")
            
            valid_appids = content_rec_df['appid'].tolist()
        else:
            valid_appids = None

        # print(f"valid_appids: {valid_appids}")

        """Collaborative Recommendations"""
        # Store recommendation scores
        scores = pd.Series(dtype=float)

        # print(played_games)

        for game in played_games:
            similar_scores = item_similarity_df[game]
            scores = scores.add(similar_scores, fill_value=0)

        # print(f"scores:\n{scores}")
        # print(f"played_games: {played_games}")

        # Remove all games the user ever played, not just liked ones
        scores = scores.drop(labels=played_games, errors="ignore")

        if valid_appids is not None:
            # filter by appids in content_rec_df
            scores = scores[scores.index.isin(valid_appids)]

        # print(f"scores:\n{scores}")
        
        # Get top N recommendations
        top_recommendations = scores.sort_values(ascending=False).head(top_n_recommendations)
        # print(top_recommendations)
        top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
        # print(top_recommendations)
        top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
        rec_df = top_recommendations[['appid','name','similarity_score']]
        rec_df = rec_df.rename(columns={'similarity_score': 'score'})

        # print(f"rec_df:\n{rec_df}")

        """Getting Results"""

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)
        if return_mrr:
            rr = get_mrr(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(rr)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    if return_full: 
        results_df['k'] = top_n_recommendations
        return results_df

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    if return_mrr:
        mean_recip_rank = np.mean(results_df['MRR'].tolist())
        print(f"Mean Reciprocal Rank: {mean_recip_rank}")
        return_list.append(mean_recip_rank)

    return tuple(return_list)

def hybrid_model_v5(
    processed_game_data, train, test, interaction_df, 
    item_similarity_df, game_details_df, similarity_method, top_n_train_examples=None, 
    top_n_recommendations=None, return_hit_rate=True, return_mean_avg_prec=True, 
    return_ndcg=True, game_threshold=5, num_content_recs=200
):
    users = list(train.keys())
    data = []
    data_cols = ['userid']
    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')

    if similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )

    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")
        
        played_games = interaction_df[interaction_df["user_steamid"] == user]["appid"].unique()

        """Collaborative Recommendations"""
        # Store recommendation scores
        scores = pd.Series(dtype=float)

        # print(played_games)

        for game in played_games:
            similar_scores = item_similarity_df[game]
            scores = scores.add(similar_scores, fill_value=0)

        # print(f"scores:\n{scores}")
        # print(f"played_games: {played_games}")

        # Remove all games the user ever played, not just liked ones
        scores = scores.drop(labels=played_games, errors="ignore")

        # print(f"scores:\n{scores}")
        
        # Get top N recommendations
        top_recommendations = scores.sort_values(ascending=False).head(num_content_recs)
        # print(top_recommendations)
        top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
        # print(top_recommendations)
        top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
        possible_rec_df = top_recommendations[['appid','name','similarity_score']]
        possible_rec_df = possible_rec_df.rename(columns={'similarity_score': 'score'})

        # print(f"possible_rec_df:\n{possible_rec_df}")

        """Content Based Recommendations"""
        # # get appids from the train set
        # train_appids = sorted(
        #     [(i, train[user][i]) for i in train[user]], 
        #     key=lambda x: x[1],
        #     reverse=True
        # )
        
        # train_appids = [int(i[0]) for i in train_appids]

        # train_appids = [i for i in train_appids if i in possible_rec_df['appid'].tolist()]

        # print(f"Num IDs: {len(train_appids)}")

        # if top_n_train_examples is not None:
        #     train_appids = train_appids[:top_n_train_examples]

        full_rec_df = content_based_recommendation(
            possible_rec_df['appid'].tolist(), # appids
            processed_game_data, # processed_game_data
            sim_df=sim_df,
            top_n=None
        )

        # top top_n_recommendations
        rec_df = full_rec_df.head(top_n_recommendations)

        # print(f"content_rec_df:\n{content_rec_df.sort_values(by='appid')}")
        
        # print(f"rec_df:\n{rec_df}")
        # assert False

        # print(f"valid_appids: {valid_appids}")
        # print(f"rec_df:\n{rec_df}")

        """Getting Results"""

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    return tuple(return_list)

def hybrid_model_v6(
    processed_game_data, train, test, interaction_df, 
    item_similarity_df, game_details_df, similarity_method, top_n_train_examples=None, 
    top_n_recommendations=None, return_hit_rate=True, return_mean_avg_prec=True, 
    return_ndcg=True, game_threshold=5, num_content_recs=200
):
    users = list(train.keys())
    data = []
    data_cols = ['userid']
    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')

    if similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )

    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")
        
        user_interactions = interaction_df[interaction_df["user_steamid"] == user]
        played_games = user_interactions["appid"].unique()

        print(f"user_interactions:\n{user_interactions}")
        print(f"played_games:\n{played_games}")
        print(f"sim_df:\n{sim_df}")
        
        content_features = user_interactions['appid'].apply(lambda x: sim_df.loc[x])
        content_features.columns = [f"cb_{i}" for i in content_features.columns]

        print(content_features)
        
        user_interactions = pd.concat([user_interactions, content_features], axis=1)

        print(user_interactions)

        collab_features = user_interactions['appid'].apply(lambda x: item_similarity_df[x])
        collab_features.columns = [f"cf_{i}" for i in collab_features.columns]

        print(collab_features)

        user_interactions = pd.concat([user_interactions, collab_features], axis=1)
        user_interactions = user_interactions.drop(columns=['playtime_forever'])

        print(user_interactions)

        assert False

        #     """Content Based Recommendations"""
        #     # get appids from the train set
        #     train_appids = sorted(
        #         [(i, train[user][i]) for i in train[user]], 
        #         key=lambda x: x[1],
        #         reverse=True
        #     )

        #     train_appids = [int(i[0]) for i in train_appids]

        #     if top_n_train_examples is not None:
        #         train_appids = train_appids[:top_n_train_examples]

        #     full_rec_df = content_based_recommendation(
        #         train_appids, # appids
        #         processed_game_data, # processed_game_data
        #         sim_df=sim_df,
        #         top_n=None
        #     )

        #     # top top_n_recommendations
        #     content_rec_df = full_rec_df.head(num_content_recs)

        #     # print(f"content_rec_df:\n{content_rec_df.sort_values(by='appid')}")
            
        #     valid_appids = content_rec_df['appid'].tolist()
        # else:
        #     valid_appids = None

        # # print(f"valid_appids: {valid_appids}")

        # """Collaborative Recommendations"""
        # # Store recommendation scores
        # scores = pd.Series(dtype=float)

        # # print(played_games)

        # for game in played_games:
        #     similar_scores = item_similarity_df[game]
        #     scores = scores.add(similar_scores, fill_value=0)

        # # print(f"scores:\n{scores}")
        # # print(f"played_games: {played_games}")

        # # Remove all games the user ever played, not just liked ones
        # scores = scores.drop(labels=played_games, errors="ignore")

        # if valid_appids is not None:
        #     # filter by appids in content_rec_df
        #     scores = scores[scores.index.isin(valid_appids)]

        # # print(f"scores:\n{scores}")
        
        # # Get top N recommendations
        # top_recommendations = scores.sort_values(ascending=False).head(top_n_recommendations)
        # # print(top_recommendations)
        # top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
        # # print(top_recommendations)
        # top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
        # rec_df = top_recommendations[['appid','name','similarity_score']]
        # rec_df = rec_df.rename(columns={'similarity_score': 'score'})

        # print(f"rec_df:\n{rec_df}")

        """Getting Results"""

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"results_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    return tuple(return_list)

def hybrid_model_v7(
    processed_game_data, train, test, interaction_df, 
    item_similarity_df, game_details_df, similarity_method='cosine', sim_df=None, 
    top_n_train_examples=None, top_n_recommendations=None, return_hit_rate=True, 
    return_mean_avg_prec=True, return_ndcg=True, return_mrr=True, game_threshold=5, num_content_recs=200,
    interaction_threshold=0.99
):
    users = list(train.keys())
    data = []
    data_cols = ['userid']

    if return_hit_rate: data_cols.append("hit")
    if return_mean_avg_prec: data_cols += ['precision@k', 'recall@k']
    if return_ndcg: data_cols.append('NDCG@k')
    if return_mrr: data_cols.append('MRR')

    if sim_df is not None and similarity_method == "cosine":
        sim_df = pd.DataFrame(
            cosine_similarity(processed_game_data), 
            columns=processed_game_data.index, 
            index=processed_game_data.index
        )
    
    interactions_per_game = interaction_df[interaction_df['interaction']==1].groupby(['appid']).size()

    # print(f"interactions_per_game:\n{interactions_per_game}")

    for idx, user in enumerate(users):
        if idx % 100 == 0:
            print(f"{idx} of {len(users)}")
        
        user_interactions = interaction_df[interaction_df["user_steamid"] == user]
        played_games = user_interactions["appid"].unique()
        marked_interactions = user_interactions[user_interactions['interaction']==1]['appid']
        games_with_low_interaction = [i for i in played_games if i not in interactions_per_game.index or interactions_per_game[i]<20]
        num_games_with_low_interaction = len(games_with_low_interaction)
        pct_low_interaction_games = num_games_with_low_interaction/len(played_games)

        use_content_based_only = False
        # if (len(marked_interactions) < game_threshold) or (pct_low_interaction_games>interaction_threshold):
        if len(marked_interactions) < game_threshold:
            # use content based filtering
            use_content_based_only = True

        """Content Based Recommendations"""
        # get appids from the train set
        train_appids = sorted(
            [(i, train[user][i]) for i in train[user]], 
            key=lambda x: x[1],
            reverse=True
        )

        train_appids = [int(i[0]) for i in train_appids]

        if top_n_train_examples is not None:
            train_appids = train_appids[:top_n_train_examples]

        full_rec_df = content_based_recommendation(
            train_appids, # appids
            processed_game_data, # processed_game_data
            sim_df=sim_df,
            top_n=None
        )

        # top top_n_recommendations
        content_rec_df = full_rec_df.head(num_content_recs)

        if not use_content_based_only:
            # print(f"content_rec_df:\n{content_rec_df.sort_values(by='appid')}")
            
            valid_appids = content_rec_df['appid'].tolist()

            # print(f"valid_appids: {valid_appids}")

            """Collaborative Recommendations"""
            # Store recommendation scores
            scores = pd.Series(dtype=float)

            # print(played_games)

            for game in played_games:
                similar_scores = item_similarity_df[game]
                scores = scores.add(similar_scores, fill_value=0)

            # print(f"scores:\n{scores}")
            # print(f"played_games: {played_games}")

            # Remove all games the user ever played, not just liked ones
            scores = scores.drop(labels=played_games, errors="ignore")

            if valid_appids is not None:
                # filter by appids in content_rec_df
                scores = scores[scores.index.isin(valid_appids)]

            # print(f"scores:\n{scores}")
            
            # Get top N recommendations
            top_recommendations = scores.sort_values(ascending=False).head(top_n_recommendations)
            # print(top_recommendations)
            top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
            # print(top_recommendations)
            top_recommendations = pd.merge(top_recommendations, game_details_df, on='appid')
            rec_df = top_recommendations[['appid','name','similarity_score']]
            rec_df = rec_df.rename(columns={'similarity_score': 'score'})
        else:
            rec_df = content_rec_df

        # print(f"rec_df:\n{rec_df}")

        """Getting Results"""

        data_lst = [user]

        # calculating metrics
        if return_hit_rate: 
            hit = recommendation_hit(user, test, rec_df)
            data_lst.append(hit)
        if return_mean_avg_prec: 
            prec_k, rec_k = get_precision_recall_at_k(user, test, rec_df, top_n_recommendations)
            data_lst += [prec_k, rec_k]
        if return_ndcg: 
            ndcg = get_ndcg(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(ndcg)
        if return_mrr:
            rr = get_mrr(user, train, test, rec_df, k=top_n_recommendations)
            data_lst.append(rr)

        data.append(tuple(data_lst))

    results_df = pd.DataFrame(data, columns=data_cols)
    
    print(f"interaction_threshold={interaction_threshold}, game_threshold={game_threshold}, num_content_recs={num_content_recs}\n\nresults_df:\n{results_df}")

    return_list = []

    if return_hit_rate:
        hit_rate = results_df[results_df['hit']].shape[0] / results_df.shape[0]
        print(f"Hit Rate: {hit_rate}")
        return_list.append(hit_rate)

    if return_mean_avg_prec:
        mean_avg_prec = np.mean(results_df['precision@k'].tolist())
        print(f"Mean Avg Precision: {mean_avg_prec}")
        return_list.append(mean_avg_prec)

    if return_ndcg:
        mean_ndcg_score = np.mean(results_df['NDCG@k'].tolist())
        print(f"Mean NDCG@k: {mean_ndcg_score}")
        return_list.append(mean_ndcg_score)

    if return_mrr:
        mean_recip_rank = np.mean(results_df['MRR'].tolist())
        print(f"Mean Reciprocal Rank: {mean_recip_rank}")
        return_list.append(mean_recip_rank)

    return tuple(return_list)

def hyperparameter_tuning_hybrid_model(weights, top_n_train_examples, top_n_recommendations):
    train, test = get_train_test()
    interaction_df, _, item_similarity_df = get_collaborative_filtering_data(train)
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    img_summary_df_v2 = pd.read_csv("../../data/top_1000_game_image_keywords.csv")
    screenshot_summary_df = pd.read_csv("../../data/top_1000_game_screenshot_summary.csv")

    game_data = process_game_data( # screenshot keywords
        game_df, 
        game_details_df, 
        img_summary_df=None, 
        img_summary_df_v2=img_summary_df_v2, 
        screenshot_summary_df=screenshot_summary_df,
        verbose=False, 
        include_image_summary=False
    )

    data = []
    for wt in weights:
        for n_train in top_n_train_examples:
            for n_recs in top_n_recommendations:
                hit_rate, mean_avg_prec, ndcg = hybrid_model(
                    game_data, 
                    train, 
                    test, 
                    interaction_df, 
                    item_similarity_df, 
                    game_details_df,
                    similarity_method='cosine', 
                    weight=wt,
                    top_n_train_examples=n_train,
                    top_n_recommendations=n_recs,
                )

                data.append((wt, n_train, n_recs, hit_rate, mean_avg_prec, ndcg))

    df = pd.DataFrame(
        data, 
        columns=['weight', 'num_train_examples', 'k', 'hit_rate', 'mean_avg_prec', 'ndcg']
    )
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"hybrid_hyperparameter_tuning_{now_str}.csv", index=False)

def evaluate_switch_hybrid_model(
        top_n_train_examples, top_n_recommendations, 
        game_thresholds, num_content_rec_lst=[], svd_k=100,
        n_iter=10, max_df=1.0, min_df=1, stop_words=None,
        return_full=False
    ):

    train, test = get_train_test()
    interaction_df, _, item_similarity_df = get_collaborative_filtering_data(train)
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    img_summary_df_v2 = pd.read_csv("../../data/top_1000_game_image_keywords.csv")
    screenshot_summary_df = pd.read_csv("../../data/top_1000_game_screenshot_summary.csv")

    game_data = process_game_data( # screenshot keywords
        game_df, 
        game_details_df, 
        img_summary_df=None, 
        img_summary_df_v2=img_summary_df_v2, 
        screenshot_summary_df=screenshot_summary_df,
        verbose=False, 
        include_image_summary=False,
        normalize_numeric=False,
        text_cols = ['header_image_summary', 'screenshot_summary'],
        svd_k=svd_k,
        svd_n_iter=n_iter,
        tfidf_max_df=max_df,
        tfidf_min_df=min_df,
        tfidf_stop_words=stop_words,
    )

    sim_df = pd.DataFrame(
        cosine_similarity(game_data), 
        columns=game_data.index, 
        index=game_data.index
    )

    data = []
    df_list = []
    for n_train in top_n_train_examples:
        for n_recs in top_n_recommendations:
            for game_threshold in game_thresholds:
                for num_content_recs in num_content_rec_lst:
                    if return_full:
                        results_df = switch_hybrid_model(
                            game_data, 
                            train, 
                            test, 
                            interaction_df, 
                            item_similarity_df, 
                            game_details_df,
                            similarity_method='cosine', 
                            sim_df=sim_df,
                            top_n_train_examples=n_train,
                            top_n_recommendations=n_recs,
                            num_content_recs=num_content_recs,
                            return_full=return_full,
                        )
                    else:
                        hit_rate, mean_avg_prec, ndcg, mrr = switch_hybrid_model(
                            game_data, 
                            train, 
                            test, 
                            interaction_df, 
                            item_similarity_df, 
                            game_details_df,
                            similarity_method='cosine', 
                            sim_df=sim_df,
                            top_n_train_examples=n_train,
                            top_n_recommendations=n_recs,
                            num_content_recs=num_content_recs,
                            return_full=return_full,
                        )
                        
                    if return_full:
                        df_list.append(results_df)
                    else:
                        data.append((game_threshold, n_train, n_recs, hit_rate, mean_avg_prec, ndcg, mrr))

    if return_full:
        df = pd.concat(df_list)
        print(df)
        df.to_csv(f"cascading_hyrbid_model_full_results.csv", index=False)
    else:
        df = pd.DataFrame(
            data, 
            columns=['game_threshold', 'num_train_examples', 'k', 'hit_rate', 'mean_avg_prec', 'ndcg', 'mrr']
        )

        print(df)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"cascading_hyrbid_model_results_{now_str}.csv", index=False)
        # df.to_csv(f"cascading_hyrbid_model_results.csv", index=False)

def evaluate_hybrid_model_v3(
        top_n_train_examples, top_n_recommendations, 
        game_thresholds, num_content_rec_lst=[]
    ):

    train, test = get_train_test()
    interaction_df, _, item_similarity_df = get_collaborative_filtering_data(train)
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    img_summary_df_v2 = pd.read_csv("../../data/top_1000_game_image_keywords.csv")
    screenshot_summary_df = pd.read_csv("../../data/top_1000_game_screenshot_summary.csv")

    game_data = process_game_data( # screenshot keywords
        game_df, 
        game_details_df, 
        img_summary_df=None, 
        img_summary_df_v2=img_summary_df_v2, 
        screenshot_summary_df=screenshot_summary_df,
        verbose=False, 
        include_image_summary=False,
        normalize_numeric=False,
        text_cols = ['header_image_summary', 'screenshot_summary']
    )

    data = []
    for n_train in top_n_train_examples:
        for n_recs in top_n_recommendations:
            for game_threshold in game_thresholds:
                for num_content_recs in num_content_rec_lst:
                    hit_rate, mean_avg_prec, ndcg = hybrid_model_v6(
                        game_data, 
                        train, 
                        test, 
                        interaction_df, 
                        item_similarity_df, 
                        game_details_df,
                        similarity_method='cosine', 
                        top_n_train_examples=n_train,
                        top_n_recommendations=n_recs,
                        num_content_recs=num_content_recs,
                    )

                    data.append((game_threshold, n_train, n_recs, hit_rate, mean_avg_prec, ndcg))

    df = pd.DataFrame(
        data, 
        columns=['game_threshold', 'num_train_examples', 'k', 'hit_rate', 'mean_avg_prec', 'ndcg']
    )

    print(df)

    df.to_csv(f"switch_hyrbid_model_results_v3.csv", index=False)

def evaluate_hybrid_model_v4(
        top_n_train_examples, top_n_recommendations, 
        game_thresholds, num_content_rec_lst=[], svd_k=100,
        n_iter=10, max_df=1.0, min_df=1
    ):

    train, test = get_train_test()
    interaction_df, _, item_similarity_df = get_collaborative_filtering_data(train)
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    img_summary_df_v2 = pd.read_csv("../../data/top_1000_game_image_keywords.csv")
    screenshot_summary_df = pd.read_csv("../../data/top_1000_game_screenshot_summary.csv")

    game_data = process_game_data( # screenshot keywords
        game_df, 
        game_details_df, 
        img_summary_df=None, 
        img_summary_df_v2=img_summary_df_v2, 
        screenshot_summary_df=screenshot_summary_df,
        verbose=False, 
        include_image_summary=False,
        normalize_numeric=False,
        text_cols = ['header_image_summary', 'screenshot_summary'],
        svd_k=svd_k,
        svd_n_iter=n_iter,
        tfidf_max_df=max_df,
        tfidf_min_df=min_df,
    )

    sim_df = pd.DataFrame(
        cosine_similarity(game_data), 
        columns=game_data.index, 
        index=game_data.index
    )

    data = []
    for n_train in top_n_train_examples:
        for n_recs in top_n_recommendations:
            for game_threshold in game_thresholds:
                for num_content_recs in num_content_rec_lst:
                    hit_rate, mean_avg_prec, ndcg, mrr = hybrid_model_v7(
                        game_data, 
                        train, 
                        test, 
                        interaction_df, 
                        item_similarity_df, 
                        game_details_df,
                        similarity_method='cosine', 
                        sim_df=sim_df,
                        top_n_train_examples=n_train,
                        top_n_recommendations=n_recs,
                        num_content_recs=num_content_recs,
                    )

                    data.append((game_threshold, n_train, n_recs, hit_rate, mean_avg_prec, ndcg, mrr))

    df = pd.DataFrame(
        data, 
        columns=['game_threshold', 'num_train_examples', 'k', 'hit_rate', 'mean_avg_prec', 'ndcg', 'mrr']
    )

    df.to_csv(f"switch_hyrbid_model_results_v3.csv", index=False)

def evaluate_hybrid_model_v5(
        top_n_train_examples, top_n_recommendations, 
        game_thresholds, num_content_rec_lst=[], svd_k=100,
        n_iter=10, max_df=1.0, min_df=1, stop_words=None,
        low_interaction_thresholds=None
    ):

    train, test = get_train_test()
    interaction_df, _, item_similarity_df = get_collaborative_filtering_data(train)
    game_df = pd.read_csv("../../data/game_player_cnt_ranked_top_1k.csv")
    game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    img_summary_df_v2 = pd.read_csv("../../data/top_1000_game_image_keywords.csv")
    screenshot_summary_df = pd.read_csv("../../data/top_1000_game_screenshot_summary.csv")

    game_data = process_game_data( # screenshot keywords
        game_df, 
        game_details_df, 
        img_summary_df=None, 
        img_summary_df_v2=img_summary_df_v2, 
        screenshot_summary_df=screenshot_summary_df,
        verbose=False, 
        include_image_summary=False,
        normalize_numeric=False,
        text_cols = ['header_image_summary', 'screenshot_summary'],
        svd_k=svd_k,
        svd_n_iter=n_iter,
        tfidf_max_df=max_df,
        tfidf_min_df=min_df,
        tfidf_stop_words=stop_words,
    )

    sim_df = pd.DataFrame(
        cosine_similarity(game_data), 
        columns=game_data.index, 
        index=game_data.index
    )

    data = []
    lists = [
        top_n_train_examples, 
        top_n_recommendations, 
        game_thresholds, 
        num_content_rec_lst,
        low_interaction_thresholds
    ]
    permutations = list(product(*lists))

    with multiprocessing.Pool(8) as pool:
        hit_rate, mean_avg_prec, ndcg, mrr = pool.starmap(
            hybrid_model_v7,
            [tuple(
                    [
                        game_data, 
                        train, 
                        test, 
                        interaction_df, 
                        item_similarity_df, 
                        game_details_df, 
                        'cosine', 
                        sim_df,
                    ] + 
                    [variables[0], variables[1]] + 
                    [True, True, True, True] + 
                    [variables[2], variables[3], variables[4]]
                )
                for variables in permutations
            ]

        )
        
        data.append((hit_rate, mean_avg_prec, ndcg, mrr))

    df = pd.DataFrame(
        data, 
        columns=['hit_rate', 'mean_avg_prec', 'ndcg', 'mrr']
    )

    print(df)

    df.to_csv(f"temp_switch_hyrbid_model_results_v3.csv", index=False)

if __name__ == "__main__":
    """Run this function to generate a CSV file with 
    results from running Offline Evaluation on several
    content-based recommendation models"""
    # content_based_offline_eval(return_full=True)

    """Run this function to generate recommendations
    for each user in our training data. These recommendations
    are saved to json files in the llm_recommendations/ folder"""
    # collect_recommendations_llm(unseen_only=True)

    """Run this function to generate a CSV file with 
    results from running Offline Evaluation on recommendations
    provided by Llama 3.1 70b model"""
    # llm_offline_eval()

    """Offline Evaluation: Collaborative Filtering"""
    # train, test = get_train_test()
    # interaction_df, user_item_matrix, item_similarity_df = get_collaborative_filtering_data(train)

    # print(f"user_item_matrix:\n{user_item_matrix}")
    # game_details_df = pd.read_csv("../../data/top_1000_game_details.csv")
    # collaborative_offline_eval_multiple_k(
    #     test, 
    #     interaction_df, 
    #     item_similarity_df, 
    #     game_details_df,
    #     return_hit_rate=True, 
    #     return_mean_avg_prec=True, 
    #     return_ndcg=True,
    #     return_full=False
    # )
    
    """Switch Hybrid Model"""
    # evaluate_switch_hybrid_model(
    #     top_n_train_examples=[None], 
    #     top_n_recommendations=[1, 5, 10, 20],
    #     game_thresholds=[5],
    #     num_content_rec_lst=[950],
    #     svd_k=10,
    #     n_iter=10,
    #     max_df=1.0,
    #     min_df=2,
    #     return_full=False,
    # )
