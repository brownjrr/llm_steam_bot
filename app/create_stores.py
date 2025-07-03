import pickle
import pandas as pd
import spacy
import re
import Levenshtein
from llm import create_review_vector_stores, get_review_retriever, get_game_data_retriever, get_reviews, get_model
from langchain_core.prompts import ChatPromptTemplate
from gensim.models import Word2Vec, KeyedVectors
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


"""Run this code to populate vectore stores. THis should only be run ONCE"""
if __name__ == "__main__":
    # get_game_data_retriever()
    # create_review_vector_stores(get_reviews())


