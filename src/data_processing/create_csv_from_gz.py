import gzip
import shutil

if __name__ == "__main__":
    for path in ["../data/top_100_game_reviews", "../data/top_1000_game_reviews"]:
        with gzip.open(path + ".gz", "rb") as f_in:
            with open(path + ".csv", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)