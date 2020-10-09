import gzip
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
import os
import re


def sentence2vec_predict(dataset, sentence2vec, idf, subset, k=10):
    predictions = [sentence2vec.most_similar_fast(dataset[i][1], idf, k) for i in subset]
    return predictions


def download_word2vec(work_dir=""):
    PATH_TO_DATA = Path(work_dir + 'data/')
    if not PATH_TO_DATA.exists():
        os.mkdir(PATH_TO_DATA)

    fr_embeddings_path = PATH_TO_DATA / 'cc.fr.300.vec.gz'
    if not fr_embeddings_path.exists():
        print("Downloading french word embeddings")
        urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz', fr_embeddings_path)
    return fr_embeddings_path


class Word2Vec():

    def __init__(self, filepath, vocab_size=50000):
        self.words, self.embeddings = self.load_wordvec(filepath, vocab_size)
        # Mappings for O(1) retrieval:
        self.word2id = {word: idx for idx, word in enumerate(self.words)}
        self.id2word = {idx: word for idx, word in enumerate(self.words)}

    def load_wordvec(self, filepath, vocab_size):
        assert str(filepath).endswith('.gz')
        words = []
        embeddings = []
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:  # Read compressed file directly
            next(f)  # Skip header
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                words.append(word)
                embeddings.append(np.fromstring(vec, sep=' '))
                if i == (vocab_size - 1):
                    break
        print('Loaded %s pretrained word vectors' % (len(words)))
        return words, np.vstack(embeddings)

    def encode(self, word):
        # Returns the 1D embedding of a given word
        if word in self.word2id:
            return self.embeddings[self.word2id[word]]
        else:  # if the word is unknown, we embed it to the 0 vector
            return np.zeros(self.embeddings[0].shape)

    def score(self, word1, word2, encode=True):
        # Return the cosine similarity: use np.dot & np.linalg.norm
        # if encode==False, we're passing the embeddings as arguments (useful for the french/english task)
        emb1 = self.encode(word1) if encode else word1
        emb2 = self.encode(word2) if encode else word2
        denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return np.dot(emb1, emb2) / denom if denom != 0 else 0

    def most_similar(self, word, k=5):
        # Returns the k most similar words: self.score & np.argsort
        similarities = [self.score(word, self.id2word[idx]) for idx in self.id2word]
        order = np.argsort(similarities)

        most_sim = []
        # can't hash with a numpy array directly
        for i in range(k):
            most_sim.append(self.id2word[order[-1 - i]])
        return most_sim


class BagOfWords():

    def __init__(self, word2vec, sentences):
        self.word2vec = word2vec
        self.sentences = sentences

    def build_idf(self, sentences):
        # build the idf dictionary: associate each word to its idf value
        # -> idf = {word: idf_value, ...}
        d = len(sentences)
        # store d to get a default value of idf if we encounter a new word
        self.d = d
        m = np.zeros(len(self.word2vec.words))
        for sentence in sentences:
            # if a word appears multiple times in a sentence, we only count it once
            seen = []
            # for word in sentence.split():
            for word in re.split('\W+', sentence):
                if word not in seen and word in self.word2vec.word2id:
                    m[self.word2vec.word2id[word]] += 1
                    seen.append(word)
        # maximum idf is log(d)
        return {word: np.log(d / max(1, m[self.word2vec.word2id[word]])) for word in self.word2vec.words}

    def encode(self, sentence, idf=None):
        # Takes a sentence as input, returns the sentence embedding
        if idf is None:
            # mean of word vectors
            return np.mean([self.word2vec.encode(word) for word in re.split('\W+', sentence)], axis=0)
        else:
            # idf-weighted mean of word vectors
            weights = np.array(
                [idf[word] if word in self.word2vec.word2id else np.log(self.d) for word in re.split('\W+', sentence)])
            return np.array([self.word2vec.encode(word) for word in re.split('\W+', sentence)]).T @ weights / np.sum(
                weights)

    def score(self, sentence1, sentence2, idf=None):
        # cosine similarity: use np.dot & np.linalg.norm
        emb1 = self.encode(sentence1, idf)
        emb2 = self.encode(sentence2, idf)
        denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return np.dot(emb1.T, emb2) / denom if denom != 0 else 0

    def most_similar(self, sentence, sentences, idf=None, k=5):
        # Return most similar sentences
        # query = self.encode(sentence, idf)
        # keys = np.vstack([self.encode(sentence, idf) for sentence in sentences])

        similarities = [self.score(sentence, sentences[idx], idf) for idx in range(len(sentences))]
        order = np.argsort(similarities)

        most_sim = []
        # can't hash with a numpy array directly
        for i in range(k):
            most_sim.append(sentences[order[-1 - i]])
        return most_sim

    def encode_sentences(self, idf=None):
        self.encoded_sentences = [self.encode(self.sentences[i], idf) for i in range(len(self.sentences))]

    def score_(self, emb1, idx, idf=None, k=5):
        emb2 = self.encoded_sentences[idx]
        denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return np.dot(emb1.T, emb2) / denom if denom != 0 else 0

    def most_similar_(self, question, idf=None, k=5):
        emb1 = self.encode(question, idf)
        similarities = [self.score_(emb1, idx, idf) for idx in range(len(self.sentences))]
        order = np.argsort(similarities)

        most_sim = []
        # can't hash with a numpy array directly
        for i in range(k):
            most_sim.append(self.sentences[order[-1 - i]])
        return most_sim

    def most_similar_fast(self, question, idf=None, k=5):
        emb1 = self.encode(question, idf)
        similarities = self.encoded_sentences @ emb1
        denom = np.linalg.norm(emb1) * np.linalg.norm(self.encoded_sentences, axis=1)
        denom = np.array([elt if elt != 0 else 1e6 for elt in denom])
        similarities = similarities / denom

        order = np.argsort(similarities)

        most_sim = []
        # can't hash with a numpy array directly
        for i in range(k):
            most_sim.append(self.sentences[order[-1 - i]])
        return most_sim
