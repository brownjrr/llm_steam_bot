import sys
import uuid
import pandas as pd
import gzip
import ast
import re
import textwrap
from io import BytesIO
from pydantic import BaseModel
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

sys.path.append("../src/llm_robert/")
from llm_game_summary import get_model


def get_reviews(app_id):
    # Open and read a .gz file
    with gzip.open(f'../data/top_100_game_reviews.gz', 'rb') as file:
        bytes = file.read()
        byte_stream = BytesIO(bytes)
        df = pd.read_csv(byte_stream)
        df = df[df['appid']==app_id]
        df = df.drop_duplicates(subset=['review'])
        df = df.sort_values(by=['weighted_vote_score'], ascending=False)

        return df['review'].tolist()
        
class SteamBotModel():
    def __init__(self, llm=get_model(), reviews=None, populate_vector_stores=False):
        self.llm = llm
        self.reviews = reviews
        self.embeddings = HuggingFaceEmbeddings()
        self.populate_vector_stores = populate_vector_stores

        # And we'll create a bound version that knows about the postal address tool
        self.game_summary = self.llm.bind_tools([ReviewSummary])
        self.tools_parser = ToolsOutputParser(pydantic_schemas=[ReviewSummary])

        # creating retrievers
        self.review_retriever = self.get_review_retriever(self.reviews)
        self.game_data_retriever = self.get_game_data_retriever()
    
    def get_review_retriever(self, reviews):
        print("Grabbing Review Retriever")

        # defining vector store
        vector_store = Chroma(
            collection_name="reviews",
            embedding_function=self.embeddings,
            persist_directory=f"../src/llm_robert/chroma_langchain_db",
        )

        if self.populate_vector_stores:
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
            uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=uuids)

        # transforming chroma vectorsotre into a retriever
        retriever = vector_store.as_retriever()

        return retriever

    def get_game_data_retriever(self):
        print("Grabbing Game Data Retriever")

        # defining vector store
        vector_store = Chroma(
            collection_name="game_data",
            embedding_function=self.embeddings,
            persist_directory=f"../src/llm_robert/chroma_langchain_db",
        )

        if self.populate_vector_stores:
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

        return retriever

    def summarize_reviews(self):
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
        
        def convert_to_json(sentence: str):
            sent = sentence.content.strip()
            return ast.literal_eval(sent)
        
        chain = {"context": self.review_retriever, "question": RunnablePassthrough()} | prompt | self.llm | RunnableLambda(convert_to_json)

        print("Invoking to get game summary")
        # Asking questions
        response = chain.invoke("Can you give a summary of the game reviews in paragraph form?")

        # print(f"response:\n{response}")

        # return response.content
        return response

    def verify(self, msg):
        print(f"Verifying message {msg}")
        if len(msg.tool_calls) <= 0:
            print("No tools were used in the chain, raising Exception")
            raise Exception("No tools were used in the chain")
        return msg

    def invoke(self, user_prompt):
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
        
        # defining chain
        app_id_chain = {"context": self.game_data_retriever, "prompt": RunnablePassthrough()} | prompt | self.llm | RunnableLambda(extract_appid)

        # Asking questions
        appid = app_id_chain.invoke(user_prompt)

        print(f"App ID: {appid}")

        # get description
        game_details_df = pd.read_csv(f"../data/top_100_game_details.csv")
        
        # filter by appid
        game_details_df = game_details_df[game_details_df['appid']==appid]

        # getting game name
        name = game_details_df['name'].values[0]

        # getting game description
        about_the_game = game_details_df['about_the_game'].values[0]

        def remove_html_tags(text):
            clean_text = re.sub(r'<.*?>', '', text)
            return clean_text
        
        about_the_game = remove_html_tags(about_the_game)

        # using LLM to summarize reviews
        review_summary = self.summarize_reviews()
        summary_str = "\n\t\t".join([f"{i}:\n\t\t\t{review_summary[i]}" for i in review_summary])

        final_message = f"""
            Game: {name}

            Description: 
            {textwrap.fill(about_the_game, width=40)}

            Review Summary: 
            {summary_str}
        """

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

def get_model():
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        model="us.meta.llama3-1-70b-instruct-v1:0", region_name="us-east-1"
    )

    return llm