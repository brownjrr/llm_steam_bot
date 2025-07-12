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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.document_loaders import CSVLoader
from langchain_core.messages import HumanMessage
import cv2
import base64
import glob


def get_model():
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        # model="us.meta.llama3-1-70b-instruct-v1:0", 
        model="us.meta.llama4-maverick-17b-instruct-v1:0",
        region_name="us-east-1"
    )

    return llm


if __name__ == "__main__":
    llm = get_model()
    data = []

    for i in glob.glob('../../data/header_images/*.jpg'):
        appid = int(i.split('\\')[-1].split('.')[0])

        # Read an image from file
        image = cv2.imread(i)

        # Encode the image as JPEG
        success, image_data = cv2.imencode('.jpg', image)

        image_data = base64.b64encode(image_data).decode("utf-8")
        
        # Create a message with the image
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe the tone of this image. Do not mention the game's name."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )

        # Invoke the model with the message
        response = llm.invoke([message])

        data.append((appid, response.content))

    df = pd.DataFrame(data, columns=['appid', 'image_summary'])

    df.to_csv("../../data/top_100_game_image_summary.csv")
