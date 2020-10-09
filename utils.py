import os
import numpy as np
from urllib.request import urlretrieve
import re
import json


def load_json(filename):
    if filename in ["train.json", "valid.json"] and not os.path.exists(filename):
        print("Downloading the dataset")
        urlretrieve("https://storage.googleapis.com/illuin/fquad/{}.zip".format(filename), filename + ".zip")
        import zipfile
        print("Extracting it")
        with zipfile.ZipFile(filename + ".zip", "r") as zip_ref:
            zip_ref.extractall("")

    if filename in ["train_squad.json", "dev_squad.json"] and not os.path.exists(filename):
        print("Downloading the dataset")
        if filename == "train_squad.json":
            urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", "train_squad.json")
            # os.rename("train-v2.0.json", "train_squad.json")
        else:
            urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", "dev_squad.json")

    if filename in ["train_squad1.json", "dev_squad1.json"] and not os.path.exists(filename):
        print("Downloading the dataset")
        if filename == "train_squad1.json":
            urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json", "train_squad1.json")
        else:
            urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json", "dev_squad1.json")

    with open(filename, encoding='utf-8') as json_file:
        data = json.load(json_file)["data"]
    return data


def create_dataset(data):
    # Crée un dataset de couples contexte-question
    dataset = [[data[theme]["paragraphs"][paragraph]["context"],
                data[theme]["paragraphs"][paragraph]["qas"][question]["question"]] for theme in range(len(data))
               for paragraph in range(len(data[theme]["paragraphs"]))
               for question in range(len(data[theme]["paragraphs"][paragraph]["qas"]))]

    unique_contexts = [data[theme]["paragraphs"][paragraph]["context"] for theme in range(len(data))
                       for paragraph in range(len(data[theme]["paragraphs"]))]

    return dataset, unique_contexts


def naive_search(question, contexts, idf, method="unweighted", k=10):
    # Classical TF-IDF: multiply number of occurences in a given document by the inverse frequency in all documents,
    # for all words in the question
    # Here: do it but for the rarest words in the question
    # question_ = question.split()[:-1]
    question = re.split('\W+', question)[:-1]
    # remove interrogative words
    to_ignore = ["Quel", "Quelle", "Quels", "Quelles", "Comment", "Qui", "Que", "Quand"]
    to_ignore += [elt.lower() for elt in to_ignore]
    question_words = [elt for elt in question if elt not in to_ignore]
    idf_question = [0 if word not in idf else idf[word] for word in question_words]

    # boost the importance of capital words and dates
    idf_question = [idf_question[i] if question_words[i].islower() else 2 * idf_question[i] for i in
                    range(len(idf_question))]
    order = np.argsort(idf_question)

    max_candidates = 4
    max_candidates = min(max_candidates, len(question_words))
    candidates = [question_words[order[-1 - i]] for i in range(max_candidates)]
    candidates_idf = [idf_question[order[-1 - i]] for i in range(max_candidates)]
    if method == "unweighted":
        scores = [np.sum([elt in context for elt in candidates]) for context in contexts]
    else:
        scores = [np.sum([context.count(candidates[i]) * candidates_idf[i] for i in range(len(candidates))]) for context
                  in contexts]

    top_k = np.argpartition(scores, -k)[-k:]  # should be faster than argsort
    order = np.argsort([scores[i] for i in top_k])

    return [contexts[top_k[order[-1 - i]]] for i in range(k)]


def naive_predict(questions, contexts, idf, subset, method="unweighted", k=10):
    return [naive_search(questions[i], contexts, idf, method, k) for i in subset]


def evaluate_retrieval(dataset, predictions, subset):
    accuracy = np.mean([dataset[subset[i]][0] == predictions[i][0] for i in range(len(subset))])
    accuracy_at_five = np.mean([dataset[subset[i]][0] in predictions[i][:5] for i in range(len(subset))])
    accuracy_at_k = np.mean([dataset[subset[i]][0] in predictions[i] for i in range(len(subset))])
    return accuracy, accuracy_at_five, accuracy_at_k


def build_idf(sentences):
    # build the idf dictionary: associate each word to its idf value
    # -> idf = {word: idf_value, ...}
    d = len(sentences)
    vocab = get_vocab(sentences)
    print(len(vocab), "unique words detected in the dataset")
    word2id = {word: idx for idx, word in enumerate(vocab)}

    m = np.zeros(len(vocab))
    for sentence in sentences:
        # if a word appears multiple times in a sentence, we only count it once
        seen = []
        for word in re.split('\W+', sentence):
            if word not in seen:
                m[word2id[word]] += 1
                seen.append(word)
    # maximum idf is log(d)
    return {word: np.log(d / max(1, m[word2id[word]])) for word in vocab}


def get_vocab(sentences):
    vocab = []
    for sentence in sentences:
        vocab += re.split('\W+', sentence)

    vocab = list(set(vocab))
    return vocab
