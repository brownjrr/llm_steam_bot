import sys
import uuid
import pandas as pd
import numpy as np
import gzip
import ast
import re
import textwrap
import json
import pickle
import Levenshtein
import textwrap
from io import BytesIO
from pydantic import BaseModel, Field
from annotated_types import Annotated
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_aws.function_calling import ToolsOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.load import dumpd, load, loads
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def get_reviews(app_id=None):
    # Open and read a .gz file
    with gzip.open(f'../data/top_100_game_reviews.gz', 'rb') as file:
        bytes = file.read()
        byte_stream = BytesIO(bytes)
        df = pd.read_csv(byte_stream)
    if app_id:
        df = df[df['appid']==app_id]
    df = df.drop_duplicates(subset=['review'])
    df = df.dropna(subset=['review'])
    df = df.sort_values(by=['weighted_vote_score'], ascending=False)

    return [(i[0], i[1]) for i in df[['appid', 'review']].values]

def get_model():
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        model="us.meta.llama3-1-70b-instruct-v1:0", region_name="us-east-1"
    )

    return llm

# Pydantic
desc_beginning = "A short description of what the reviews say about"

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

class SteamBotModel():
    def __init__(self, llm=None, reviews=None, populate_vector_stores=False):
        self.llm = llm
    
    def summarize_reviews(self, appid):
        # Defining prompt template
        template = """Use the following context from game reviews, answer the 
        question (Respond in JSON with `Target Audience`, `Graphics`, `Quality`,
        `Requirements`, `Difficulty`, `Game Time/Length`, `Story`, `Bugs`, 
        `Other Features`, `Sentiment` keys):
        {context}

        Question: {question}
        """

        # Creating prompt
        prompt = ChatPromptTemplate.from_template(template)
        
        def process_game_summary_review(game_summary_review) -> str:
            ret_str = ""
            
            for i in game_summary_review.__fields__.keys():
                content = '\n\t\t\t  '.join(textwrap.wrap(getattr(game_summary_review, i), width=100))
                ret_str += f"\t{i.upper().replace('_', ' ')}:\n\t\t\t* {content}\n\t"
            
            return ret_str
        
        retriever = get_review_retriever(get_reviews(), skip_populating=True, filter_app_id=str(appid))

        chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | self.llm.with_structured_output(GameReviewSummary) | RunnableLambda(process_game_summary_review)

        print(f"Invoking to get game review summary (appid={appid})")

        # Asking questions
        # response = chain.invoke("Can you give a summary of the game reviews in paragraph form?", filter={"source": appid})
        response = chain.invoke("Can you give a summary of the game reviews in paragraph form?")

        return response

    def verify(self, msg):
        print(f"Verifying message {msg}")
        if len(msg.tool_calls) <= 0:
            print("No tools were used in the chain, raising Exception")
            raise Exception("No tools were used in the chain")
        return msg

    def invoke(self, user_prompt):
        """Finding name of game referenced by user"""
        template = """
            Get the name of the game mentioned in the following prompt:

            Prompt: {prompt}

            Return the results as a single Python string, with no other text
        """

        # Creating prompt
        prompt = ChatPromptTemplate.from_template(template)

        def extract_name_str(ai_message):
            return re.sub(r'[^\w\s]', '', ai_message.content.strip().lower())

        # defining chain
        game_name_chain = {"prompt": RunnablePassthrough()} | prompt | self.llm | RunnableLambda(extract_name_str)

        # Asking questions
        game_name = game_name_chain.invoke(user_prompt)

        # find the most similar game in our dataset
        games_df = pd.read_csv(f"../data/top_100_games.csv")

        print(games_df)

        games_df['similarity_score'] = games_df['name'].apply(lambda x: Levenshtein.ratio(game_name, re.sub(r'[^\w\s]', '', x.lower())))

        sim_df = games_df[['appid', 'name', 'similarity_score']].sort_values(
            by=['similarity_score'], 
            ascending=False
        )

        print(sim_df)
        
        if sim_df['similarity_score'].values[0] > 0.7:
            appid = sim_df['appid'].values[0]
        else:
            appid = None

        print(f"App ID: {appid}")

        game_details_df = pd.read_csv(f"../data/top_100_game_details.csv")
        
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
        review_summary = self.summarize_reviews(appid)

        # wrapping description
        about_the_game = "\n\t\t".join(textwrap.wrap("\t"+about_the_game, width=100))

        final_message = f'''
        **Game:** {name}
        
        **Description:**
        {about_the_game}

        **Review Summary:**
        {review_summary}
        '''

        return final_message

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
    print("Grabbing Review Retriever")

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
    if filter_app_id:
        retriever = vector_store.as_retriever()
    else:
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
        persist_directory=f"./chroma_langchain_db",
    )

    if not skip_populating:
        # Create documents from df records
        loader = CSVLoader(
            file_path="../data/top_100_game_details.csv", 
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
    
    chain = {"context": get_review_retriever(get_reviews(), skip_populating=True), "question": RunnablePassthrough()} | prompt | llm | RunnableLambda(convert_to_json)

    # saving chain
    chain_dict = dumpd(chain)

    with open("./saved_chains/review_summary_chain.json", "w+") as f:
        json.dump(chain_dict, f)
