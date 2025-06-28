import gzip
import pandas as pd
import chromadb
from io import BytesIO
from uuid import uuid4
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_reviews(app_id):
    # Open and read a .gz file
    with gzip.open('../../data/top_100_game_reviews.gz', 'rb') as file:
        bytes = file.read()
        byte_stream = BytesIO(bytes)
        df = pd.read_csv(byte_stream)
        df = df[df['appid']==app_id]
        df = df.drop_duplicates(subset=['review'])
        df = df.sort_values(by=['weighted_vote_score'], ascending=False)

        return df['review'].tolist()

def get_review_retriever(reviews, populate_vector_store=False):
    print("Grabbing Retriever")

    # create embedding
    embeddings = HuggingFaceEmbeddings()

    # defining vector store
    vector_store = Chroma(
        collection_name="reviews",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
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

def get_model():
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        model="us.meta.llama3-1-70b-instruct-v1:0", region_name="us-east-1"
    )

    return llm

def summarize_reviews(reviews, llm=None, populate_vector_store=False):
    # creating retriever
    retriever = get_review_retriever(
        reviews, 
        populate_vector_store=populate_vector_store
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
    
    given the following context from game reviews:
    {context}

    Question: {question}
    """

    # Creating prompt
    prompt = ChatPromptTemplate.from_template(template)

    # defining chain
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    # Asking questions
    response = chain.invoke("Can you give a summary of the game reviews in paragraph form?")

    print(response.content)

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

# DEPRECATED
# def summarize_reviews(reviews, model=get_model()):
#     prompt = ChatPromptTemplate.from_template(
#         "Given the following list of lists of game reviews, return a paragraph summarizing the reviews: {reviews}"
#     )

#     # Now we can create a chain, and in this case I want to get that output formatted correctly
#     chain = prompt | model

#     response = chain.invoke([{"reviews": reviews}])

#     print(response)


if __name__ == "__main__":
    # getting reviews for appid=550
    review_list = get_reviews(550)

    # defining LLM
    llm = get_model()

    # Change this to True if this your first 
    # time running the summarize_reviews() function
    populate_vector_store=False

    # using LLM to summarize reviews
    summarize_reviews(review_list, llm=llm, populate_vector_store=populate_vector_store)
