# Models Package Importing 
from keras.models import load_model
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import pickle
w2v_model = Word2Vec.load('C:\\Users\\Peter Samoaa\\Downloads\\results\\model.w2v')
model = load_model('C:\\Users\\Peter Samoaa\\Downloads\\results\\model.h5')
with open('C:\\Users\\Peter Samoaa\\Downloads\\results\\tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('C:\\Users\\Peter Samoaa\\Downloads\\results\\encoder.pkl', 'rb') as handle:
    encoder = pickle.load(handle)

# Setting 
# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# Senti lable, score methods
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE
def predict(text, include_neutral=True):
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)
    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}
#===================apply model on tweets===========================================#
import pandas as pd 
df = pd.read_csv("D:\Proccessed Data\GeoSentiTweet\Deep Learning\Implicit Coordinates\Apple_LSTM.csv")
df['score'] =0.0
df['Sentiment_Class']= ''
df['Sentiment_Approach'] = 'DeepLearning_WordEmb_LSTM'
df['Sentiment_lable'] = 0
for i in range (len(df)):
    try:
        results = predict(df.Tweet[i])
        df.score[i]= results['score']
        df.Sentiment_Class[i] = results['label']
    except:
        df.score[i]= 0.0
        df.Sentiment_Class[i] = 'NEUTRAL'
    print ("process {} out of {}".format(i,len(df)))


for i in range (len(df)):
    if df.Sentiment_Class[i] == 'NEGATIVE':
        df['Sentiment_lable'][i] = -1
    elif df.Sentiment_Class[i]== 'NEUTRAL':
        df['Sentiment_lable'][i] = 0
    else:
        df['Sentiment_lable'][i] = 1
