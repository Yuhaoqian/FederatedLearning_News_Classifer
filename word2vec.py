from keras.layers import Input, Dense, Flatten, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential, Model
import lab_util
import numpy as np
class Word2VecModel():
    def __init__(self, vocab_size, embed_dim, hidden_size, context_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.hidden_size = hidden_size
    def build_model(self,):
        Inputs = Input(shape=(self.context_size * 2,))
        EmbeddingL = Embedding(input_dim=self.vocab_size,
                               output_dim=self.embed_dim,
                               input_length=self.context_size * 2)(Inputs)

        Hidden = Flatten()(EmbeddingL)
        Hidden = Dense(self.hidden_size, activation='relu')(Hidden)
        Hidden = Dense(self.hidden_size, activation='relu')(Hidden)
        Output = Dense(self.vocab_size, activation='softmax')(Hidden)

        self.Model_Pre = Model(inputs=Inputs, outputs=Output)
        self.Model_Pre.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        self.Model_embed = Model(inputs=Inputs, outputs=EmbeddingL)

    def TrainModel(self, X, Y, n_batch, n_epochs):
        self.Model_Pre.fit(X, Y, batch_size=n_batch, epochs=n_epochs)

def get_w2v_model(tokenized_corpus, vocab_size, embed_dim, hidden_size, context_size, n_batch, n_epochs):
    ngrams = lab_util.get_ngrams(tokenized_corpus, context_size)
    # convert to Numpy arrange of X and Y
    X = []
    Y = []
    for i in range(len(ngrams)):
        X.append(ngrams[i][0])
        Y.append(ngrams[i][1])
    X = np.stack(X)
    Y = np.stack(Y)
    Y = to_categorical(Y, num_classes=vocab_size)
    # build model
    MODEL = Word2VecModel(vocab_size, embed_dim, hidden_size, context_size)
    MODEL.build_model()
    print(MODEL.Model_Pre.summary())
    # train model
    MODEL.TrainModel(X, Y, n_batch, n_epochs)
    return MODEL.Model_Pre

def learn_reps_word2vec(corpus, vocab_size, embed_dim, hidden_size, context_size, n_batch, n_epochs):
    tokenizer = lab_util.Tokenizer()
    tokenizer.fit(corpus)
    tokenized_corpus = tokenizer.tokenize(corpus)

    ngrams = lab_util.get_ngrams(tokenized_corpus, context_size)

    # convert to Numpy arrange of X and Y
    X = []
    Y = []
    for i in range(len(ngrams)):
        X.append(ngrams[i][0])
        Y.append(ngrams[i][1])
    X = np.stack(X)
    Y = np.stack(Y)

    Y = to_categorical(Y, num_classes=tokenizer.vocab_size)

    # build model
    MODEL = Word2VecModel(tokenizer.vocab_size, embed_dim, hidden_size, context_size)
    MODEL.build_model()
    print(MODEL.Model_Pre.summary())

    # train model
    MODEL.TrainModel(X, Y, n_batch, n_epochs)

    # get matrix of word embedding
    Word_embedding = MODEL.Model_embed.get_weights()[0]
    # MODEL.Model_Pre.save("whole.h5")
    # MODEL.Model_embed.save("embedding.h5")
    return Word_embedding