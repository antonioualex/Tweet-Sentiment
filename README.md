# **Sentiment Analysis on tweets**
<p>
A program that performs <b>Sentiment Analysis</b> on tweets based on a Neural Network model.
Analysis can be processed through two ways. Either a fast or a precise one.
</p>

## <em>**Implementation**</e>


### **Get tweet data based on hashtags**
<p>Setting up a connection to tweeter API using tweepy package. Utilize Streaming protocol to deliver data through our tweepy API connection with new results being sent through that connection whenever new matches occur.
</p>

### **Data Preproccess**
<p>
In order to proccess our tweet posts we need to:    <ul><li>Replace special tokens where it is needed to. (E.g. url)
<li>Tokenize each post.
<li>Set unknown tokens for words that are not contained in our word to index dictionary.
<li>Replace them with integers in order for the keras model to be able to predict sentiments.
<li>If precise is set to False , Padding will be applied on the smaller tweets so all of them have the same size and run faster.</ul></p>

### **Load Sentiment model**
 <p>Loading sentiment model on our preproccessed data and extracting results (full_text, sentiment, created_at, tweet_id) to a csv file.</p>

## ***Usage***
<p>You can run the programm by pressing in the command line python tweet_sentiment_analysis.py. If you want to change default arguments you can press python tweet_sentiment_analysis.py -h to see a full list of possible arguments. <b>(E.g To run the script with precision you can press in command line python tweet_sentiment_analysis.py --precision=True)</b>. Below you can see the full list of possible arguments.</p>


>optional ***arguments***: <br>
  &nbsp;&nbsp;&nbsp;&nbsp;  **-h**, **--help**      show this help message and exit <br>
  &nbsp;&nbsp;&nbsp;&nbsp;  **--precision**     Get the sentiments for each tweet separately if set to True, else for all the tweets at once. Precision &nbsp;&nbsp;&nbsp;&nbsp;default value is set to False.<br>
  &nbsp;&nbsp;&nbsp;&nbsp; **--num-tweets**      Set the number of tweet posts for each API request. Default value is set to 250.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;    **--hashtags**      Comma separated list of hashtags to search for, note that at least one of the hashtags must be contained in a tweet in order to &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;be fetched.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;  **--model-file**    The path of the keras model. Root is the current directory.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;  **--output-file**   The path of the results file. Root is the current directory.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;  **--auth-file**     The path of the file containing the authentication credentials in order to connect to Twitter API. Root is the current directory.<br>
  &nbsp;&nbsp;&nbsp;&nbsp; **--w2i-file**      The path of the file containing the mapping of words and integers. Root is the current directory.<br>
