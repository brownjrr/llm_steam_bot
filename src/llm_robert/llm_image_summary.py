import pandas as pd
import cv2
import base64
import glob
import os
import ast
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.messages import HumanMessage


def get_model():
    print("Grabbing Model")

    # creating model
    llm = ChatBedrockConverse(
        model="us.meta.llama4-maverick-17b-instruct-v1:0",
        # model="us.meta.llama3-1-70b-instruct-v1:0",
        region_name="us-east-1"
    )

    return llm

def get_header_image_tone(llm):
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
                {
                    "type": "text", 
                    "text": "Describe the tone of this image. Be concise. Do not mention the game's name."
                },
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

    df.to_csv("../../data/top_1000_game_image_summary.csv")

def get_header_image_keywords(llm):
    data = []
    files = glob.glob('../../data/header_images/*.jpg')
    for idx, i in enumerate(files):
        if idx % 10 == 0:
            print(f"{idx} of {len(files)}")
            
        appid = int(i.split('\\')[-1].split('.')[0])

        # Read an image from file
        image = cv2.imread(i)

        # Encode the image as JPEG
        success, image_data = cv2.imencode('.jpg', image)

        image_data = base64.b64encode(image_data).decode("utf-8")
        
        # Create a message with the image
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "Mention a few keywords that best describe this image. Return results as a comma-separated list with no other words."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )

        # Invoke the model with the message
        response = llm.invoke([message])

        data.append((appid, response.content))

    df = pd.DataFrame(data, columns=['appid', 'image_keywords'])

    df.to_csv("../../data/top_1000_game_image_keywords.csv")

def get_image_url_for_imagecaptionloader(id, df):
    # collect images
    img_urls = ast.literal_eval(df['screenshots'].tolist()[0])
    urls = []
    for i in img_urls:
        url_path_key = "path_full" if "path_full" in i else "path_thumbnail"
        url = i[url_path_key]
        urls.append(url)
    return urls

def get_images_as_bytes(id):
    data_list = []

    for i in glob.glob(f"../../data/screenshots/{id}/*.jpg"):
        # Read an image from file
        image = cv2.imread(i)

        # Encode the image as JPEG
        success, image_data = cv2.imencode('.jpg', image)
        
        if success:
            image_data = base64.b64encode(image_data).decode("utf-8")
            data_list.append(image_data)
        else:
            assert False, "could not convert image"

    return data_list

def get_screenshot_summary(llm):
    df = pd.read_csv("../../data/top_1000_game_details.csv")

    def get_screenshot_summary_helper(id):
        screenshot_dir = f"../../data/screenshots/{id}"

        if not os.path.isdir(screenshot_dir):
            return None
        
        images = get_images_as_bytes(id)
        images = [
            {
                "type": f"image_url", 
                f"image_url": {"url": f"data:image/jpeg;base64,{byte_str}"}
            }
            for byte_str in images
        ]

        key_words = set()
        for image in images:
            # Create a message with the image
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": "Mention a few keywords that best describe this image. Return results as a comma-separated list with no other words."
                    },
                    image
                ],
            )

            # Invoke the model with the message
            response = llm.invoke([message])
            temp_key_words = {i.strip() for i in response.content.split(",")}
            key_words |= temp_key_words
        
        assert len(key_words)>0, f"Couldn't find keywords for {id}"

        return key_words

    df['screenshot_summary'] = df['appid'].apply(get_screenshot_summary_helper)
    
    print(df)

    df.to_csv("../../data/top_1000_game_screenshot_summary.csv")


if __name__ == "__main__":
    llm = get_model()

    # creating version 1 of header image summarizer
    # get_header_image_tone(llm)

    # creating version 2 of header image summarizer
    # get_header_image_keywords(llm)

    # creating version 1 of screenshot image summarizer
    # get_screenshot_summary(llm)
