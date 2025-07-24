import gzip
import pandas as pd
from io import BytesIO
from uuid import uuid4
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader

def get_reviews(app_id, relative_path_to_base="../../"):
    # Open and read a .gz file
    with gzip.open(f'{relative_path_to_base}data/top_100_game_reviews.gz', 'rb') as file:
        bytes = file.read()
        byte_stream = BytesIO(bytes)
        df = pd.read_csv(byte_stream)
        df = df[df['appid']==app_id]
        df = df.drop_duplicates(subset=['review'])
        df = df.sort_values(by=['weighted_vote_score'], ascending=False)

        return df['review'].tolist()

def get_review_retriever(reviews, populate_vector_store=False, relative_path_llm_dir="./"):
    print("Grabbing Retriever")

    # create embedding
    embeddings = HuggingFaceEmbeddings()

    # defining vector store
    vector_store = Chroma(
        collection_name="reviews",
        embedding_function=embeddings,
        persist_directory=f"{relative_path_llm_dir}chroma_langchain_db",
    )

    if populate_vector_store:
        documents = []

        # Create documents from reviews
        for idx, content in enumerate(reviews):
            documents.append(
                Document(
                    page_content=content,
                    id=idx,
                )
            )

        # adding documents to vector store
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)

    # transforming chroma vectorsotre into a retriever
    retriever = vector_store.as_retriever()

    return retriever

def get_game_data_retriever(populate_vector_store=False, relative_path_llm_dir="./"):
    print("Grabbing Retriever")

    # create embedding
    embeddings = HuggingFaceEmbeddings()

    # defining vector store
    vector_store = Chroma(
        collection_name="game_data",
        embedding_function=embeddings,
        persist_directory=f"{relative_path_llm_dir}chroma_langchain_db",
    )

    if populate_vector_store:
        # Create documents from df records
        loader = CSVLoader(
            file_path="../../data/top_100_game_details.csv", 
            encoding='utf-8',
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': ['appid', 'name', 'about_the_game']
            }
        )

        documents = loader.load()

        # adding documents to vector store
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)

    # transforming chroma vectorsotre into a retriever
    retriever = vector_store.as_retriever()

    return retriever

def get_model():
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        model="us.meta.llama3-1-70b-instruct-v1:0", region_name="us-east-1"
    )

    return llm

def summarize_reviews(reviews, llm=None, populate_vector_store=False, relative_path_llm_dir="./"):
    # creating retriever
    retriever = get_review_retriever(
        reviews, 
        populate_vector_store=populate_vector_store,
        relative_path_llm_dir=relative_path_llm_dir
    )

    print("Defining Prompt Template")

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

    # defining chain
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    # Asking questions
    response = chain.invoke("Can you give a summary of the game reviews in paragraph form?")

    return response.content

"""Combining reviews for each game into a single document"""
def combine_reviews():
    # Open and read a .gz file
    with gzip.open('../../data/top_100_game_reviews.gz', 'rb') as file:
        bytes = file.read()
        byte_stream = BytesIO(bytes)
        df = pd.read_csv(byte_stream)
    
    game_df = pd.read_csv("../../data/top_100_game_details.csv")
    
    print(game_df)
    print(game_df.columns)

    # FIXME: COMPLETE

def extract_appid(ai_message):
    return int(ai_message.content.strip())

def invoke_llm(model, prompt_str, populate_vector_store=False, relative_path_llm_dir="./", relative_path_to_base="../../"):
    # creating retriever
    retriever = get_game_data_retriever(
        populate_vector_store=populate_vector_store,
        relative_path_llm_dir=relative_path_llm_dir
    )

    """Finding App ID of game referenced by user"""
    template = """
        Return the appid of the game mentioned in the following prompt, given the following context. Return answer as a single number and no other text:
        {context}

        Prompt: {prompt}
    """

    # Creating prompt
    prompt = ChatPromptTemplate.from_template(template)

    # defining chain
    app_id_chain = {"context": retriever, "prompt": RunnablePassthrough()} | prompt | model | RunnableLambda(extract_appid)

    # Asking questions
    appid = app_id_chain.invoke(prompt_str)

    # print(f"App ID: {appid}")

    # get description
    game_details_df = pd.read_csv(f"{relative_path_to_base}data/top_100_game_details.csv")
    
    # filter by appid
    game_details_df = game_details_df[game_details_df['appid']==appid]

    # getting game name
    name = game_details_df['name'].values[0]

    # getting game description
    about_the_game = game_details_df['about_the_game'].values[0]

    # print(f"about_the_game:\n{about_the_game}")

    # getting reviews for appid
    review_list = get_reviews(appid, relative_path_to_base)

    # Change this to True if this your first 
    # time running the summarize_reviews() function
    populate_vector_store=False

    # using LLM to summarize reviews
    review_summary = summarize_reviews(
        review_list, 
        llm=model, 
        populate_vector_store=populate_vector_store, 
        relative_path_llm_dir=relative_path_llm_dir
    )

    print(f"review_summary:\n{review_summary}")

    # final_message = f"""
    #     Game: {name}

    #     Description: 
    #     {about_the_game}

    #     Review Summary: 
    #     {review_summary}
    # """

    final_message = f"""
        Game: {name}

        Description: 
        {about_the_game}
    """

    return final_message


if __name__ == "__main__":
    # # getting reviews for appid=550
    # review_list = get_reviews(550)

    # # defining LLM
    # llm = get_model()

    # # Change this to True if this your first 
    # # time running the summarize_reviews() function
    # populate_vector_store=False

    # # using LLM to summarize reviews
    # summarize_reviews(review_list, llm=llm, populate_vector_store=populate_vector_store)
    
    llm = get_model()

    # prompt = "Tell me more about the reception of Left 4 Dead."
    prompt = "Tell me more about the reception of Left 4 Dead 2."

    invoke_llm(llm, prompt, populate_vector_store=False)

    # df= pd.read_csv("../../data/top_100_games.csv")
    # print(df.drop_duplicates(subset=['name']))
