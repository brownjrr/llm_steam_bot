import pandas as pd
import numpy as np
import glob
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def process_game_data(game_df, details_df):
    # Merge game data with details
    game_df = game_df.merge(details_df, left_on='appid', right_on='id', how='inner')

    # filter out any records where type != 'game'
    game_df = game_df[game_df['type']=='game']

    # removing any games that haven't come out yet
    game_df['coming_soon'] = game_df['release_date'].apply(lambda x: ast.literal_eval(x)['coming_soon'])
    game_df = game_df[game_df['coming_soon']==False]

    # creating release_year feature
    game_df['release_year'] = game_df['release_date'].apply(
        lambda x: pd.to_datetime(
            ast.literal_eval(x)['date'], errors='coerce'
        ).year
    )

    # dropping records with no release year
    game_df = game_df.dropna(subset=['release_year'])

    # processing recommendations column
    def process_recommendations(x):
        if not isinstance(x, str) and np.isnan(x):
            return None
        else:
            return ast.literal_eval(x)['total']
    game_df['recommendations'] = game_df['recommendations'].apply(process_recommendations)

    # processing the developers column
    def process_developers(x):
        if isinstance(x, str):
            x = ast.literal_eval(x)
            return ", ".join(x)
        else:
            return None
    game_df['developers'] = game_df['developers'].apply(process_developers)

    # processing name column
    def process_name(x):
        if isinstance(x, str):
            return x
        else:
            return None
    game_df['name_x'] = game_df['name_x'].apply(process_name) 

    # preprocessing about_the_game column
    def process_about_the_game(x):
        if isinstance(x, str):
            return re.sub(r"<.*?>", '', x)
        else:
            return None
    game_df['about_the_game'] = game_df['about_the_game'].apply(process_about_the_game) 

    # processing categories column
    def process_categories(x):
        if isinstance(x, str):
            categories = [i['description'] for i in ast.literal_eval(x)]
            return ", ".join(categories)
        else:
            return None
    game_df['categories'] = game_df['categories'].apply(process_categories)
    
    # preprocessing genres column
    def process_genres(x):
        if isinstance(x, str):
            genres = [i['description'] for i in ast.literal_eval(x)]
            return ", ".join(genres)
        else:
            return None
    game_df['genres'] = game_df['genres'].apply(process_genres)

    # preprocessing publishers column 
    def process_publishers(x):
        if isinstance(x, str):
            publishers = ast.literal_eval(x)
            return ", ".join(publishers)
        else:
            return None
    game_df['publishers'] = game_df['publishers'].apply(process_publishers)

    text_cols = [
        'name_x',
        'about_the_game',
        'developers',
        'categories',
        'genres',
        'publishers',
    ]

    numeric_cols = [
        'is_free',
        # 'ratings', # FIXME: handling this requires some thought. Exclude for now
        'recommendations',
        'required_age',
        'release_year'
    ]

    def preprocess_text_column(x):
        # impute missing data
        x = "" if x is None else x

        # cast to lowercase
        try:
            x = x.lower()
        except:
            print(x)
            assert False

        # remove non-alphanumeric characters
        x = re.sub(r'[^a-zA-Z0-9 \']', ' ', x)

        return x

    # clean text columns
    for col in text_cols:
        game_df[col] = game_df[col].apply(preprocess_text_column)

    # combine text columns
    game_df['text'] = game_df[text_cols[0]]
    for col in text_cols[1:]:
        game_df['text'] += " " + game_df[col]

    # vectorize text column
    vectorizer = TfidfVectorizer()
    text_vec_df = pd.DataFrame(vectorizer.fit_transform(game_df['text']).toarray())

    # concatenate text_vec_df with numeric cols
    final_game_df = pd.DataFrame(
        data=np.hstack([game_df[numeric_cols], text_vec_df]),
        columns=numeric_cols+list(text_vec_df.columns)
    )
    
    print(final_game_df)
    print(final_game_df.shape)

    return final_game_df

    

if __name__ == "__main__":
    """Processing Game Data"""
    # game_df = pd.read_csv("../../data/raw_game_data.csv")
    # game_details_df = pd.read_csv("../../data/game_details_sample.csv")
    # process_game_data(game_df, game_details_df)

    
    """(TESTING) Counting the number of files in each folder"""
    # successful = glob.glob("../../data/successful_requests/*")
    # failed = glob.glob("../../data/failed_requests/*")
    # no_data = glob.glob("../../data/no_data_requests/*")

    # print(f"successful: {len(successful)}")
    # print(f"failed: {len(failed)}")
    # print(f"no_data: {len(no_data)}")