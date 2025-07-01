import pickle
import pandas as pd
import spacy
import re
import Levenshtein
from llm import get_review_retriever, get_game_data_retriever, get_reviews, get_model
from langchain_core.prompts import ChatPromptTemplate
from gensim.models import Word2Vec, KeyedVectors


"""Run this code to populate vectore stores. THis should only be run ONCE"""
if __name__ == "__main__":
    get_game_data_retriever()
    get_review_retriever(get_reviews())