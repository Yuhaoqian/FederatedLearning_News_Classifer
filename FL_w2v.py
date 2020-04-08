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
# print(f"Read {len(data)} total news article.")

# for i in range(10):
#     train_texts = data['short_description'].iloc[i*10000:(i+1)*10000]
#     corpus = train_texts
#     tokenized_corpus = Tokenizer.tokenize(corpus)
#     current_model = word2vec.get_w2v_model(tokenized_corpus, vocab_size, embed_dim=500, hidden_size=16, context_size=5, n_batch=500, n_epochs=3)
#     current_model.save('saved_model/model' + str(i + 1) + '.h5')
weights_of_10_silos = []
for i in range(10):
    model = load_model('saved_model/model' + str(i + 1) + '.h5')
    weights = model.get_weights()
    weights_of_10_silos.append(weights)

Aggregated_weights = []
for idx_list, list in enumerate(weights_of_10_silos):
    if idx_list == 0:
        for w in list:
            Aggregated_weights.append(np.zeros(w.shape, dtype='float32'))
    for idx_w, w in enumerate(list):
        Aggregated_weights[idx_w] += w

for w in Aggregated_weights:
    w /= 10

w2v = word2vec.Word2VecModel(vocab_size, embed_dim=500, hidden_size=16, context_size=5)
w2v.build_model()
w2v.Model_Pre.set_weights(Aggregated_weights)

w2v.Model_Pre.save('saved_model/FederatedLearning_Model.h5')
reps_word2vec = Aggregated_weights[0]
words = ["she", "trump", "love"]
show_tokens = [vectorizer.tokenizer.word_to_token[word] for word in words]
lab_util.show_similar_words(vectorizer.tokenizer, reps_word2vec, show_tokens)







