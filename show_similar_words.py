import lab_util
import pandas as pd
import numpy as np
np.random.seed(0)
np.set_printoptions(suppress=True)
# my python module
import word2vec
from keras.models import load_model

data = pd.read_json('News_Category_Dataset.json', lines='True')
data = data.sample(frac=1)  # shuffle
train_texts = data['short_description'].iloc[0:100000]
corpus = train_texts
Tokenizer = lab_util.Tokenizer()
Tokenizer.fit(corpus)
vocab_size = Tokenizer.vocab_size
vectorizer = lab_util.CountVectorizer()
vectorizer.fit(corpus)


model = load_model('saved_model/FederatedLearning_Model.h5')
reps_word2vec = model.get_weights()[0]
words = ["she", "trump", "love"]
show_tokens = [vectorizer.tokenizer.word_to_token[word] for word in words]
lab_util.show_similar_words(vectorizer.tokenizer, reps_word2vec, show_tokens)