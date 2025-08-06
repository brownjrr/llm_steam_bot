import pandas as pd

# Get recommendation for user
def recommend_games_for_user(user_id, interaction, user_item_matrix, item_similarity_df, df_top_n_game_details, df_users_owned_games, show_output=False, top_n=5):
    # Get the user’s interaction vector
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found.")
        return []
    # Filter played used games
    played_games = interaction[interaction["user_steamid"] == user_id]["appid"].unique()
    # Store recommendation scores
    scores = pd.Series(dtype=float)
    for game in played_games:
        similar_scores = item_similarity_df[game]
        scores = scores.add(similar_scores, fill_value=0)
    # Remove all games the user ever played, not just liked ones
    scores = scores.drop(labels=played_games, errors="ignore")
    # Get top N recommendations
    top_recommendations = scores.sort_values(ascending=False).head(top_n)
    top_recommendations = pd.DataFrame(top_recommendations).reset_index().rename(columns={0: 'similarity_score'})
    top_recommendations = pd.merge(top_recommendations, df_top_n_game_details, on='appid')
    top_recommendations = top_recommendations[['appid','name','similarity_score']]

    if show_output==True:
      ### Prints for visualization
      print(f'-----------------------\nUser: {user_id}')
      # Print played games
      games = list(df_users_owned_games[df_users_owned_games['user_steamid']==user_id]['parsed_owned_games'])[0]
      games_sorted = sorted(games, key=lambda x: x['playtime_forever'], reverse=True)
      print('---\nUser most played games:')
      for game in games_sorted[:5]:
          print(f"- {game['name']}")
      # Print Recommendations
      print("---\nTop Recommended Games:")
      print(top_recommendations)
      print("\n")
    pass
    return top_recommendations