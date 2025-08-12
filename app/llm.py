import sys
import uuid
import pandas as pd
import gzip
import ast
import re
import textwrap
import json
import pickle
import os
from io import BytesIO
from pydantic import BaseModel, Field
from annotated_types import Annotated
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.load import dumpd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool, InjectedToolArg
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

sys.path.append("../src/data_processing/")
sys.path.append("../src/recommender/")

from games import process_game_data
#from collaborative_memory_based import recommend_games_for_user 

df_memory_based_recommendations = pd.read_csv("../data/df_memory_based_recommendations.csv")

GAME_NOT_FOUND_STR = "<GAME_NOT_FOUND>"

# content_sim_df


def content_based_recommendation(appids, X, sim_df=None, similarity_method=None, top_n=10):
    if similarity_method is not None and similarity_method not in ['cosine']:
        assert False, "This function is not capable of handling this similarity method"

    if sim_df is None:
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
    game_df = pd.read_csv("../data/game_player_cnt_ranked_top_1k.csv")[['appid', 'name']]
    app_similarities = app_similarities.merge(game_df, on=['appid'], how='left')

    # separate app row from suggestion rows
    app_rows = app_similarities[app_similarities['appid'].isin(appids)]
    app_similarities = app_similarities[~app_similarities['appid'].isin(appids)]
    if top_n is not None: app_similarities = app_similarities[:top_n]

    if app_similarities['score'].isna().all():
       app_similarities['score'] = len(app_similarities) - app_similarities.index

    return app_similarities

def get_reviews(app_id=None):
    # Open and read a .gz file
    with gzip.open(f'../data/top_1000_game_reviews.gz', 'rb') as file:
        bytes = file.read()
        byte_stream = BytesIO(bytes)
        df = pd.read_csv(byte_stream)
    if app_id:
        df = df[df['appid']==app_id]
    df = df.drop_duplicates(subset=['review'])
    df = df.dropna(subset=['review'])
    df = df.sort_values(by=['weighted_vote_score'], ascending=False)

    return [(i[0], i[1]) for i in df[['appid', 'review']].values]

def get_model(temp=None, top_p=None):
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        # model="us.meta.llama3-1-70b-instruct-v1:0", region_name="us-east-1"
        model="us.meta.llama4-maverick-17b-instruct-v1:0",
        region_name="us-east-1"
    )

    return llm

def summarize_reviews(appid, llm):
    # Defining prompt template
    template = """Use the following context from game reviews, answer the 
    question (Respond in JSON with several different and unique thoughts about the `Target Audience`, `Graphics`, `Quality`,
    `Requirements`, `Difficulty`, `Game Time/Length`, `Story`, `Bugs`, 
    `Other Features`, `Sentiment` keys that each have something different and interesting to say):
    {context}

    Question: {question}
    """

    # Creating prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    def process_game_summary_review(game_summary_review) -> str:
        ret_str = ""
        emojis = ["🎯", "🎨", "✅", "💻", "🎮", "⏱️", "📚", "🐞", "💬"]
        for index, key in enumerate(game_summary_review.__fields__.keys()):

            content = textwrap.fill(getattr(game_summary_review, key), width=100)
            ret_str += f"**{emojis[index]} {key.upper().replace('_', ' ')}**  \n* {content}\n\n"
        
        return ret_str
    
    retriever = get_review_retriever(skip_populating=True, filter_app_id=str(appid))

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm.with_structured_output(GameReviewSummary) | RunnableLambda(process_game_summary_review)

    print(f"Invoking to get game review summary (appid={appid})")

    # Asking questions
    response = chain.invoke("Can you give a summary of the game reviews in paragraph form?")

    return response
    
@tool("get_game_info")
def get_game_info(appid: int):
    """
    A tool which returns information related to a video game on Steam, including a
    summary of its Steam reviews.

    Examples:
    1. If the user asks for information about the game "Left 4 Dead 2" and you find this game's 
    appid to be 550, you would specify the appid as 550
    2. If the user says "Tell me more about Left 4 Dead 2" and you find Left 4 Dead 2 
    to have an appid of 550, you would specify the appid as 550
    3. If the user says "Tell me about the reception of Left 4 Dead 2" and you find Left 4 Dead 2 
    to have an appid of 550, you would specify the appid as 550
    """
    game_details_df = pd.read_csv(f"../data/top_1000_game_details.csv")
    
    # filter by appid
    game_details_df = game_details_df[game_details_df['appid']==appid]

    # getting game name
    name = game_details_df['name'].values[0]

    # getting game description
    about_the_game = game_details_df['about_the_game'].values[0]

    def remove_html_tags(text):
        clean_text = re.sub(r'<.*?>', '', text).strip()
        return clean_text
    
    about_the_game = remove_html_tags(about_the_game)

    # using LLM to summarize reviews
    review_summary = summarize_reviews(appid, get_model())

    # wrapping description
    about_the_game = "\n".join(textwrap.wrap(about_the_game, width=100))

    final_message = f"## 🎮 **Game**\n{str(name)}\n\n---\n\n" + \
        f"## 📖 **Description**\n{about_the_game}\n\n---\n\n" + \
        f"## 📝 **Review Summary**\n\n{review_summary}"

    return final_message

@tool("get_game_recommendation")
def get_game_recommendation(appids) -> str: 
    """
    A tool which returns recommendations based on a given video game.

    Examples:
    1. If the user asks for recommendations similar to the game "Left 4 Dead 2" 
    and you find this game's appid to be 550, you would specify appids as [550]
    2. If the user asks for recommendations similar to the games "Left 4 Dead 2" 
    and "Dota 2" and you find this game's appids to be 550 and 570, respectively, 
    you would specify appids as [550, 570]
    """

    if isinstance(appids, str): appids = ast.literal_eval(appids)
    
    df = pd.read_csv("../data/processed_game_data.csv")
    game_details_df = pd.read_csv("../data/top_1000_game_details.csv")
    sim_df = pd.read_csv("../data/game_content_similarity.csv")
    sim_df = sim_df.set_index("appid")
    sim_df.columns = [int(i) for i in sim_df.columns]

    num_games = 5
    recs_df = content_based_recommendation(appids, df, sim_df=sim_df, top_n=num_games)
    game_str = ", ".join([str(game_details_df[game_details_df['appid'] == appid]['name'].values[0]) for appid in appids])

    final_message = f"""## 🎮 Similar Games to {game_str}

Based on our analysis, here are {num_games} games that are most similar to {game_str}. These recommendations are ranked by similarity score.

| Game Title | Similarity Score |
|------------|------------------|
"""

    for _, row in recs_df.iterrows():
        final_message += f"| {row['name']} | {row['score']:.3f} |\n"

    return final_message

# Pydantic
desc_beginning = "A few unique sentences of what the reviews say about"

class GameReviewSummary(BaseModel):
    """Game Review Summary to return to user"""
    audience: str = Field(description="A short description of the target audience for this game.")
    graphics: str = Field(description=f"{desc_beginning} the game's graphics")
    quality: str = Field(description=f"{desc_beginning} the game's quality")
    requirements: str = Field(description=f"{desc_beginning} the game's system requirements")
    difficulty: str = Field(description=f"{desc_beginning} the game's difficulty")
    game_length: str = Field(description=f"{desc_beginning} the game's length")
    story: str = Field(description=f"{desc_beginning} the game's story")
    bugs: str = Field(description=f"{desc_beginning} bugs in the game")
    # other_features: str = Field(description=f"{desc_beginning} any other features")
    sentiment: str = Field(description=f"A short description of the overall sentiment of the game, based on reviews")

class UserRecommendationAnswer(BaseModel):
    answer: str = Field(description="Answer to the question: Yes or No")
    explanation: str = Field(description="Reason for answering Yes or No to the question")

class SteamBotModel():
    def __init__(self, llm=None, reviews=None, populate_vector_stores=False):
        self.llm = llm
    
    def verify(self, msg):
        print(f"Verifying message {msg}")
        if len(msg.tool_calls) <= 0:
            print("No tools were used in the chain, raising Exception")
            raise Exception("No tools were used in the chain")
        return msg

    def invoke(self, user_profile_id, user_prompt):
        """Finding name of game referenced by user"""

        template = """
            Return the name of the game(s) mentioned in the following prompt:

            Prompt: {prompt}

            Return as List of Strings or an empty list (e.g. []), with no other text. Only use double quotes (") around Strings.
        """
        
        def load_json_str(ai_message):
            print(f"game name chain ai_message:\n{ai_message.content}")
            return json.loads(ai_message.content.replace("```json", "").replace("```", ""))
        
        retriever = get_game_data_retriever(skip_populating=True)

        game_name_chain = (
            {"prompt": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | self.llm 
            | RunnableLambda(load_json_str)
        )

        def load_app_json_str(ai_message):
            print(f"load_app_json_str ai_message:\n{ai_message.content}")
            
            if GAME_NOT_FOUND_STR in ai_message.content:
                return GAME_NOT_FOUND_STR
            
            return json.loads(ai_message.content.replace("```json", "").replace("```", ""))
        
        template = "Find the appid of the following game in our corpus, given the following context:\n\nGame: '{game}'\n\nContext:{context}" \
                "\n\nReturn results as a JSON dictionary with keys `name` and `appid`. " \
                f"If you don't think this game exists in our corpus, return the following string: '{GAME_NOT_FOUND_STR}'.\n " \
                "If it seems the user wants recommendations based on a profile, return an empty Python list (e.g. [])\n\n" \
                "DO NOT return python code."
        
        appid_chain = (
            {"context": retriever, "game": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | RunnableLambda(load_app_json_str)
        )

        # Asking questions
        response = (game_name_chain | appid_chain.map()).invoke(user_prompt)

        print(f"response:\n{response}")
        
        # Error Handling: Could Not Find Game(s)
        if isinstance(response, list) and len(response)>=1 and all([i == GAME_NOT_FOUND_STR for i in response]):
            similar_game_chain = (
                {"context": retriever, "game": RunnablePassthrough()}
                | ChatPromptTemplate.from_template(
                    "Return answer as a comma-separated list (0-5 items)--no other text--of the most similar games " \
                    "(by name) to the following game, given the following context:\n\n" \
                    "Game Name: {game}\n\nContext: {context}"
                )
                | self.llm
            )

            response = (game_name_chain | similar_game_chain.map()).invoke(user_prompt)
            
            similar_game_list = []
            for item in response:
                similar_game_list += [i.strip() for i in item.content.strip().split(",")]

            similar_game_list = ''.join(f'- **{game}**\n' for game in similar_game_list)
            
            return "## 🎮 Game Not Found\n" \
            "We couldn't find the game you were looking for based on your prompt.\n\n" \
            "Here are some games we do have that may be similar to your prompt:\n\n---\n\n" \
            f"{similar_game_list}"
        
        # User Profile Recommendations
        if len(response) == 0:
            profile_recs_chain = (
                ChatPromptTemplate.from_template(
                    "Is the following prompt seeking recommendations either for themselves or for a third party profile?" \
                    "Return your answer as a JSON dictionary with keys `answer` which takes on 'Yes' " \
                    "or 'No' and `explanation` which is the explanation for your answer.\n\n" \
                    "Prompt: {prompt}"
                )
                | self.llm.with_structured_output(UserRecommendationAnswer)
            )

            profile_rec_response = profile_recs_chain.invoke(user_prompt)
            print(f"profile_rec_response:\n{profile_rec_response}")

            if profile_rec_response.answer == "Yes":
                user_id_chain = (
                    ChatPromptTemplate.from_template(
                        "Extract the user id (with no other text) from the following prompt. If no user id" \
                        f"is found, return the following string: '{user_profile_id}'. \n\n" \
                        "Prompt: {prompt}"
                    )
                    | self.llm
                )

                user_id_response = user_id_chain.invoke(user_prompt).content.strip()
                print(f"user_id_response:\n{user_id_response}")

                if user_id_response != "None":
                    agent = create_react_agent(self.llm, [get_user_recommendation])
                    system_message = {"role": "system", "content": f"User is asking about the profile with user_steamid={user_id_response}"}
                    input_message = {"role": "user", "content": user_prompt}
                    agent_response = agent.invoke({
                        "messages": [system_message, input_message],
                        "user_steamid": int(user_id_response)
                    }, {"recursion_limit": 100})

                    print(f"agent_response:\n{agent_response}")

                    return [msg for msg in agent_response['messages'] if isinstance(msg, ToolMessage)][-1].content
                else:
                    df = pd.read_csv("../data/game_player_cnt_ranked_top_1k.csv").head(10)

                    final_message = "I don't have your gaming history, so here are some popular " \
                    "Steam games\n\n## 🎮 Popular Games\n" \
                    "| Game Title | Player Count |\n" \
                    "|------------|------------------|\n"

                    for _, row in df.iterrows():
                        final_message += f"| {row['name']} | {row['player_count']:,} |\n"

                    # return "If you select a profile, I can give you recommendations based on your" \
                    # " gaming history. Otherwise, give me some examples of games you like and I can" \
                    # "come up with a few recommendations."
                    return final_message

        # Getting game recommendations or game review summary
        appids = [str(game['appid']) for game in response]
        appids_str = ", ".join(appids)
        agent = create_react_agent(self.llm, [get_game_info, get_game_recommendation])
        system_message = {
            "role": "system", 
            "content": f"The user mentions the following game(s) with appid(s): " \
            f"{appids_str}. If the user asks for recommendations, run get_game_recommendation()." \
            "If the users asks for info about a game, run get_game_info()."
        }
        input_message = {"role": "user", "content": user_prompt}
        agent_response = agent.invoke({
            "messages": [system_message, input_message],
        }, {"recursion_limit": 100})

        # Ensure only the last ToolMessage is included in agent_response['messages']
        agent_response['messages'] = [i for i in agent_response['messages'] if isinstance(i, ToolMessage)][-1:]

        return next(
                    (msg.content for msg in agent_response['messages'] if isinstance(msg, ToolMessage)),
                    "Please ask about either a game summary or for recommendations similar to a game"
                )

class ReviewSummary(BaseModel):
    """
    Summary of the reviews for a game
    """
    target_audience: Annotated[str, "A short summary of the intended audience of the game."]
    graphics: Annotated[str, "A short summary of the sentiment on the game's graphics."]
    requirements: Annotated[str, "A short summary of the system requirements needed to run this game."]
    difficulty: Annotated[str, "A short summary of the difficulty of the game"]
    game_length: Annotated[str, "A short summary of the length of the game"]
    story: Annotated[str, "A short summary of the sentiment on the game's story"]
    bugs: Annotated[str, "A short summary of the sentiment of the bugginess of the game"]
    other: Annotated[str, "A short summary of any other important important information offered by the reviews"]
    overall_sentiment: Annotated[str, "A short summary of the overall sentiment of the game"]


def get_review_retriever(reviews, skip_populating=False, filter_app_id=None):
    print(f"Grabbing Review Retriever for App ID: {filter_app_id}")

    embeddings = HuggingFaceEmbeddings()

    # defining vector store
    vector_store = Chroma(
        collection_name="reviews",
        embedding_function=embeddings,
        persist_directory=f"./chroma_langchain_db/{filter_app_id}/",
    )

    if not skip_populating:
        # Create documents from reviews
        for idx, (appid, content) in enumerate(reviews):
            if idx % 5000 == 0:
                print(idx)

            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000, chunk_overlap=50
            )

            recursive_splits = recursive_splitter.split_text(content)

            # an empty review is causing documents to be empty which throws an error
            if not recursive_splits:
                continue

            documents = []

            for chunk_num, chunk in enumerate(recursive_splits):
                documents.append(
                    Document(
                        page_content=content,
                        id=idx,
                        metadata={"source": appid, "chunk_num": chunk_num, "review_id":idx},
                    )
                )
            
            # adding documents to vector store
            uuids = [str(uuid.uuid4()) for _ in range(len(documents))]

            print(f"adding documents [idx={idx}]")
            vector_store.add_documents(documents=documents, ids=uuids)

    # transforming chroma vectorsotre into a retriever
    retriever = vector_store.as_retriever()

    print("Finished Grabbing Review Retriever")

    return retriever

def get_game_data_retriever(skip_populating=False):
    print("Grabbing Game Data Retriever")

    embeddings = HuggingFaceEmbeddings()

    # defining vector store
    vector_store = Chroma(
        collection_name="game_data",
        embedding_function=embeddings,
        persist_directory=f"./chroma_langchain_db/game_data",
    )

    if not skip_populating:
        # Create documents from df records
        loader = CSVLoader(
            file_path="../data/top_1000_game_details.csv", 
            encoding='utf-8',
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': ['appid', 'name', 'about_the_game']
            }
        )

        documents = loader.load()

        # adding documents to vector store
        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)

    # transforming chroma vectorsotre into a retriever
    retriever = vector_store.as_retriever()

    print("Finished Grabbing Game Data Retriever")

    return retriever

def create_review_vector_stores(reviews):
    print("Grabbing Review Retriever")

    embeddings = HuggingFaceEmbeddings()

    # grouping reviews by appid
    get_app_review_dict = {}
    for i in reviews:
        if i[0] in get_app_review_dict:
            get_app_review_dict[i[0]].append(i[1])
        else:
            get_app_review_dict[i[0]] = [i[1]]
    
    # Get all subdirectories under ./chroma_langchain_db/
    chroma_db_dir = "./chroma_langchain_db/"
    subdirs = [d for d in os.listdir(chroma_db_dir) if os.path.isdir(os.path.join(chroma_db_dir, d))]
    print("Subdirectories under ./chroma_langchain_db/:", subdirs)
    
    # for appid, reviews in get_app_review_dict.items():
    #     print(f"APPID: {appid}")

    #     # defining vector store
    #     vector_store = Chroma(
    #         collection_name="reviews",
    #         embedding_function=embeddings,
    #         persist_directory=f"./chroma_langchain_db/{appid}/",
    #     )

    #     for idx, content in enumerate(reviews):
    #         if idx % 5000 == 0:
    #             print(idx)

    #         doc = Document(
    #             page_content=content,
    #             id=idx,
    #             metadata={"source": appid},
    #         )

    #         documents = [doc]

    #         # adding documents to vector store
    #         uuids = [str(uuid.uuid4()) for _ in range(len(documents))]

    #         print(f"adding documents [idx={idx}]")
    #         vector_store.add_documents(documents=documents, ids=uuids)

def create_and_save_app_id_chain(llm):
    """Finding App ID of game referenced by user"""
    template = """
        Return the appid of the game mentioned in the following prompt, given the following context. Return answer as a single number and no other text:
        {context}

        Prompt: {prompt}
    """

    # Creating prompt
    prompt = ChatPromptTemplate.from_template(template)

    def extract_appid(ai_message):
        return int(ai_message.content.strip())
    
    retriever = get_game_data_retriever()

    with open("./saved_chains/rgame_data_retriever.pkl", "wb+") as f:
        pickle.dump(retriever, f)

    # defining chain
    app_id_chain = {"context": retriever, "prompt": RunnablePassthrough()} | prompt | llm | RunnableLambda(extract_appid)

    # # Save to a pickel file
    # with open("./saved_chains/app_id_chain.pkl", "wb+") as f:
    #     pickle.dump(app_id_chain, f)

    # saving chain
    # app_id_chain_dict = dumpd(app_id_chain)

    # with open("./saved_chains/app_id_chain.json", "w+") as f:
    #     json.dump(app_id_chain_dict, f)

def create_and_save_review_summary_chain(llm):
    # Defining prompt template
    template = """For each summary asked for, provide information (separated in paragraphs) about 

    - Target Audience
    - Graphics
    - Quality
    - Requirements
    - Difficulty
    - Game Time/Length
    - Story
    - Bugs
    - Other Features
    - Overall Sentiment
    
    given the following context from game reviews (return result as a dictionary with where the required sections are keys and the summary text as values):
    {context}

    Question: {question}
    """

    # Creating prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    def convert_to_json(sentence: str):
        sent = sentence.content.strip()
        return ast.literal_eval(sent)
    
    chain = {"context": get_review_retriever(skip_populating=True), "question": RunnablePassthrough()} | prompt | llm | RunnableLambda(convert_to_json)

    # saving chain
    chain_dict = dumpd(chain)

    with open("./saved_chains/review_summary_chain.json", "w+") as f:
        json.dump(chain_dict, f)

@tool("get_user_recommendation")
def get_user_recommendation(user_steamid: Annotated[int, InjectedToolArg]):
    """
    Returns personalized game recommendations based on a user's Steam ID.

    Examples:
    1. Run this function if the user asks for recommendations for themselves or for
    another user. Don't provide any arguments to this function.
    """
    print(f"RUNNING get_user_recommendation(user_steamid:{user_steamid}, {type(user_steamid)})")
    try:
    #     # Load pre calculated memory based data from CSVs
    #     user_item_matrix = pd.read_csv("../data/memory_based_user_item_matrix_1000_games.csv", index_col=0)
    #     item_similarity_df = pd.read_csv("../data/df_memory_based_item_similarity_1000_games.csv", index_col=0)
    #     interaction_df = pd.read_csv("../data/df_memory_based_interaction_1000_games.csv")
    #     df_top_n_game_details = pd.read_csv("../data/top_1000_game_details.csv")
    #     df_users_owned_games = pd.read_csv("../data/df_users_owned_games.csv")

    #     # format issues
    #     if user_item_matrix.index.dtype in ['float64', 'float32']:
    #         user_item_matrix.index = user_item_matrix.index.astype("int64")
    #     if user_item_matrix.columns.dtype in ['float64', 'float32']:
    #         user_item_matrix.columns = user_item_matrix.columns.astype("int64")
    #     if item_similarity_df.index.dtype in ['float64', 'float32']:
    #         item_similarity_df.index = item_similarity_df.index.astype("int64")
    #     if item_similarity_df.columns.dtype in ['float64', 'float32']:
    #         item_similarity_df.columns = item_similarity_df.columns.astype("int64")
    #     item_similarity_df.columns = item_similarity_df.columns.astype(int)
    #     item_similarity_df.index = item_similarity_df.index.astype(int)

        # Call recommender function
        # recommendations = recommend_games_for_user(
        #     user_id=user_steamid,
        #     interaction=interaction_df,
        #     user_item_matrix=user_item_matrix,
        #     item_similarity_df=item_similarity_df,
        #     df_top_n_game_details=df_top_n_game_details,
        #     df_users_owned_games=df_users_owned_games,
        #     show_output=False,
        #     top_n=10
        # )
        recommendations = df_memory_based_recommendations[df_memory_based_recommendations['user_id']==user_steamid]
        print(f"recommendations:\n{recommendations}")

        if len(recommendations) == 0:
            return f"Sorry, no recommendations found for Steam user ID `{user_steamid}`."

        output = f"## 🎮 Recommendations for Steam Profile `{user_steamid}`\n\n"
        output += "| Game Title | Similarity Score |\n|------------|------------------|\n"

        for _, row in recommendations.iterrows():
            output += f"| {row['name']} | {row['similarity_score']:.3f} |\n"

        return output

    except Exception as e:
        print(f"Error: {e}")
        return f"Something went wrong fetching recommendations for Steam ID `{user_steamid}`"
