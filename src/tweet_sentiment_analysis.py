import argparse
from datetime import datetime, timedelta
from timeit import repeat
import moment
import pandas as pd
import keras
import tweepy
import json
import re
import os
import time
from pprint import pprint
from nltk.tokenize import TweetTokenizer


class MyStreamListener(tweepy.StreamListener):
    """Implementation of a Stream listener class so API can get continious data.

    Init with the number of tweet posts we want our listener to wait for.

    Args:
        tweepy.StreamListener (type): .

    Returns:
        List: A List containing a tweet.
    """

    def __init__(self, num_tweets):
        self.num_tweets = num_tweets
        self.tweets = []
        super(MyStreamListener, self).__init__()

    def on_status(self, tweet):
        if len(self.tweets) < self.num_tweets:
            if ("RT @" not in tweet.text) and (not tweet.retweeted):
                text = (
                    tweet.extended_tweet["full_text"]
                    if hasattr(tweet, "extended_tweet")
                    else tweet.text
                )
                self.tweets.append(
                    {
                        "full_text": text,
                        "created_at": tweet.created_at,
                        "tweet_id": tweet.id,
                    }
                )
        else:
            return False

    def on_error(self, error):
        print("efages timeout mwri polla requests kane pause")
        time.sleep(60)
        return True


def _replace_special_tokens(sequence):
    """Replace special tokens using regex.

    This function replaces special tokens in the given string if any exist. The
    special tokens are urls, tweet usernames, hashtags etc.

    Args:
        sequence (str): A string value to apply the transformations on.

    Returns:
        str: The input value without special tokens.
    """
    sequence = re.sub(
        """http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]
        [0-9a-fA-F]))+""",
        "<url>",
        sequence,
    )
    sequence = re.sub("@\S+", "<user>", sequence)
    sequence = re.sub("#\S+", "<hashtag>", sequence)
    sequence = re.sub(":-?\)", "<smileface>", sequence)
    sequence = re.sub(":-?D", "<lolface>", sequence)
    sequence = re.sub(":\|", "<neutralface>", sequence)
    sequence = re.sub(":\(", "<sadface>", sequence)
    sequence = re.sub("([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))", "<number>", sequence)
    return sequence


def _preprocess_tokens(df):
    """Apply preprocessing on a dataframe.

    Remove special tokens from the `full_text` column of the input dataframe
    and then tokenize it using the `TweetTokenizer` class.

    Args:
        df (pandas.DataFrame): The pandas dataframe containing tweets.

    Returns:
        pandas.DataFrame: The processed dataframe.
    """
    tt = TweetTokenizer(preserve_case=False)
    df["full_text_processed"] = df["full_text"].apply(_replace_special_tokens)
    df["full_text_processed"] = df["full_text_processed"].apply(
        lambda x: tt.tokenize(x)
    )
    return df


def _set_unknown_tokens(df, w2i_dict):
    """Replace unknown words.

    There are words where the keras model is not trained on. Replace these
    words with the `*#*UNK*#*` token.

    Args:
        df (pandas.DataFrame): The pandas dataframe containing tweets.

    Returns:
        pandas.DataFrame: The processed dataframe.
    """
    df["full_text_processed"] = df["full_text_processed"].apply(
        lambda x: [word if word in w2i_dict else "*#*UNK*#*" for word in x]
    )
    return df


def _w2i(df, w2i_dict, precise=False):
    """Replace words with integers.

    Given a dictionary of word to index replace the words of the input
    dataframe with integers in order for the keras model to be able to predict
    sentiments. Padding will be applied on the smaller tweets if precise is set
    to False.

    Args:
        df (pandas.DataFrame): The pandas dataframe containing tweets.
        w2i_dict (dict): Word to index dictionary
        precise (bool): If set to False padding will be applied in order to
        bring all the tweets at the same length.

    Returns:
        pandas.DataFrame: The processed dataframe.
    """
    if precise:
        # Specific token for each post.
        df["full_text_processed"] = df["full_text_processed"].apply(
            lambda x: [w2i_dict.get(i) for i in x]
        )
    else:
        # Making all tweet tokens same size by filling them with the PAD token.
        max = df["full_text_processed"].apply(len).max()
        df["full_text_processed"] = df["full_text_processed"].apply(
            lambda x: ((max - len(x)) * ["*#*PAD*#*"] + x)
        )
        df["full_text_processed"] = df["full_text_processed"].apply(
            lambda x: [w2i_dict.get(i) for i in x]
        )
    return df


def _set_up_api(config):
    """Initialize a tweet API given the appropriate tokens.

    Returns:
        api: A tweet API.
    """

    auth = tweepy.OAuthHandler(config["consumer_key"], config["consumer_secret"])
    auth.set_access_token(config["access_key"], config["access_secret"])
    api = tweepy.API(auth, wait_on_rate_limit=True)

    return api


def _score_tweets(**kwargs):
    """Main function to create sentiment analysis on tweeter posts.


    Initializing a connection to API and adding a listener to it.
    Optimizing tweets and adapt them to a sentiment analysis model.
    Extracting results to a csv. Based on given arguments or the default ones.

    Args:
        df (pandas.DataFrame): The pandas dataframe containing tweets.
        w2i_dict (dict): Word to index dictionary
        precise (bool): If set to False padding will be applied in order to
        bring all the tweets at the same length.

    Returns:
        pandas.DataFrame: The processed dataframe.
    """

    precision = eval(kwargs["precision"])
    hashtags = kwargs["hashtags"]
    sentiment_model_path = kwargs["model_file"]
    auth_path = kwargs["auth_file"]
    w2i_path = kwargs["w2i_file"]
    output_path = kwargs["output_file"]
    num_tweets = kwargs["num_tweets"]

    try:
        with open(auth_path, "r") as f:
            auth_config = json.load(f)
    except OSError as e:
        print(f"Incorrect path direction for auth.json file. \n{e}")
        exit()

    try:
        sentiment_model = keras.models.load_model(sentiment_model_path)
    except OSError as e:
        print(f"Incorrect path direction for keras model. \n{e}")
        exit()

    api = _set_up_api(auth_config)

    while True:

        tweets_listener = MyStreamListener(num_tweets)
        stream = tweepy.Stream(api.auth, tweets_listener)
        stream.filter(track=[hashtags], languages=["en"])
        tweets = tweets_listener.tweets

        df = pd.DataFrame(tweets)

        # Words to integer
        w2i_dict = {}
        try:
            with open(w2i_path) as w2i:
                w2i_dict = json.loads(w2i.read())

        except OSError as e:
            print(f"Incorrect path direction for w2i.json file. \n{e}")
            exit()

        df = _preprocess_tokens(df)
        df = _set_unknown_tokens(df, w2i_dict)
        df = _w2i(df, w2i_dict, precision)

        df["sentiment"] = (
            df["full_text_processed"].apply(
                lambda x: sentiment_model.predict([x])[0][0]
            )
            if precision
            else sentiment_model.predict(df["full_text_processed"].tolist())
        )

        head = False if os.path.exists(output_path) else True
        df.to_csv(
            output_path,
            columns=["full_text", "sentiment", "created_at", "tweet_id"],
            header=head,
            index=False,
            mode="a",
        )
        print(f"\nInserting {num_tweets} rows into csv...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tweets",
        help="""Set the number of tweet posts for each API request.
        Default value is set to 250.""",
        default=250,
        metavar="",
        type=int,
    )
    parser.add_argument(
        "--precision",
        help="""Get the sentiments for each tweet separately if set to True,
        else for all the tweets at once. Precision default value is set to False.""",
        default="False",
        metavar="",
        type=str,
    )
    parser.add_argument(
        "--hashtags",
        help="""Comma separated list of hashtags to search for, note that
        at least one of the hashtags must be contained in a tweet in order
        to be fetched.""",
        default="#trump,#Trump,#biden,#Biden,#elections",
        metavar="",
        type=str,
    )
    parser.add_argument(
        "--model-file",
        help="""The path of the keras model. Root is the current directory.""",
        default=os.path.join(os.getcwd(), "model.h5"),
        metavar="",
        type=str,
    )
    parser.add_argument(
        "--output-file",
        help="""The path of the results file. Root is the current directory.""",
        default=os.path.join(os.getcwd(), "results.csv"),
        metavar="",
        type=str,
    )
    parser.add_argument(
        "--auth-file",
        help="""The path of the file containing the authentication credentials
        in order to connect to Twitter API. Root is the current directory.""",
        default=os.path.join(os.getcwd(), "auth.json"),
        metavar="",
        type=str,
    )
    parser.add_argument(
        "--w2i-file",
        help="""The path of the file containing the mapping of words and
        integers. Root is the current directory.""",
        default=os.path.join(os.getcwd(), "w2i.json"),
        metavar="",
        type=str,
    )

    args = parser.parse_args()
    arguments = args.__dict__
    _score_tweets(**arguments)
