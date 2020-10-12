import os
import numpy as np
import re
import json
from urllib.request import urlretrieve


def load_json(dataset, split, data_dir="data/"):
    # load the json file corresponding to the dataset and split chosen
    filename = split + "_" + dataset + ".json"
    if not os.path.exists(data_dir + filename):
        if filename in ["train_fquad.json", "valid_fquad.json"]:
            print("Downloading the dataset")
            urlretrieve("https://storage.googleapis.com/illuin/fquad/{}.zip".format(split + ".json"),
                        data_dir + filename + ".zip")
            import zipfile
            print("Extracting it")
            with zipfile.ZipFile(data_dir + filename + ".zip", "r") as zip_ref:
                zip_ref.extractall(data_dir)
            os.rename(data_dir + split + ".json", data_dir + filename)

        elif filename in ["train_squad.json", "valid_squad.json"]:
            print("Downloading the dataset")
            if filename == "train_squad.json":
                urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
                            data_dir + "train_squad.json")
            else:
                urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
                            data_dir + "valid_squad.json")

        elif filename in ["train_squad1.json", "valid_squad1.json"]:
            print("Downloading the dataset")
            if filename == "train_squad1.json":
                urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
                            data_dir + "train_squad1.json")
            else:
                urlretrieve("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
                            data_dir + "valid_squad1.json")

        else:
            raise Exception(
                "Test dataset not found. Please name it test_<dataset_name>.json and place it in the data_dir directory")

    with open(data_dir + filename, encoding='utf-8') as json_file:
        data = json.load(json_file)["data"]
    return data


def create_dataset(data):
    # Create a dataset of tuples context-question, and also return the unique contexts present in the dataset
    dataset = [[data[theme]["paragraphs"][paragraph]["context"],
                data[theme]["paragraphs"][paragraph]["qas"][question]["question"]] for theme in range(len(data))
               for paragraph in range(len(data[theme]["paragraphs"]))
               for question in range(len(data[theme]["paragraphs"][paragraph]["qas"]))]

    unique_contexts = [data[theme]["paragraphs"][paragraph]["context"] for theme in range(len(data))
                       for paragraph in range(len(data[theme]["paragraphs"]))]

    return dataset, unique_contexts


def naive_search(question, contexts, idf, method="unweighted", k=10):
    # Classical TF-IDF: multiply the number of occurences in a given document by the inverse frequency in all documents,
    # for all words in the question
    # Here: do it but for the rarest words in the question. The unweighted method just checks if a given word appears
    # in a document and thus doesn't even use the term "tf"
    # It is a very naive method but it looks fairly powerful : around 80% Acc@20 score
    question = re.split('\W+', question)[:-1]
    # remove interrogative words (we could do the same for the English ones, it doesn't actually affect the performance
    # dramatically)
    to_ignore = ["Quel", "Quelle", "Quels", "Quelles", "Comment", "Qui", "Que", "Quand"]
    to_ignore += [elt.lower() for elt in to_ignore]
    question_words = [elt for elt in question if elt not in to_ignore]
    idf_question = [0 if word not in idf else idf[word] for word in question_words]

    # boost the importance of capital words and dates. Again, it doesn't change the performance by much
    idf_question = [idf_question[i] if question_words[i].islower() else 2 * idf_question[i] for i in
                    range(len(idf_question))]
    order = np.argsort(idf_question)

    # This is a "hyperparameter", setting it to 3-5 looks optimal here, we could make it a function of the number of
    # words in the sentence
    max_candidates = 4
    max_candidates = min(max_candidates, len(question_words))
    candidates = [question_words[order[-1 - i]] for i in range(max_candidates)]
    candidates_idf = [idf_question[order[-1 - i]] for i in range(max_candidates)]
    # most simple method but outperforms the weighted one
    if method == "unweighted":
        scores = [np.sum([elt in context for elt in candidates]) for context in contexts]
    else:
        scores = [np.sum([context.count(candidates[i]) * candidates_idf[i] for i in range(len(candidates))]) for context
                  in contexts]

    top_k = np.argpartition(scores, -k)[-k:]  # should be faster than argsort
    order = np.argsort([scores[i] for i in top_k])

    return [contexts[top_k[order[-1 - i]]] for i in range(k)]


def naive_predict(questions, contexts, idf, subset, method="unweighted", k=10):
    # Return the k top contexts for each question indexed by elements of the subset list
    return [naive_search(questions[i], contexts, idf, method, k) for i in subset]


def evaluate_retrieval(dataset, predictions, subset):
    # Return the accuracy scores of the prediction made on subset
    accuracy = np.mean([dataset[subset[i]][0] == predictions[i][0] for i in range(len(subset))])
    accuracy_at_five = np.mean([dataset[subset[i]][0] in predictions[i][:5] for i in range(len(subset))])
    accuracy_at_k = np.mean([dataset[subset[i]][0] in predictions[i] for i in range(len(subset))])
    return accuracy, accuracy_at_five, accuracy_at_k


def build_idf(sentences):
    # build the idf dictionary: associate each word to its idf value
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
    # create a list of the unique words present in the corpus
    vocab = []
    for sentence in sentences:
        vocab += re.split('\W+', sentence)

    vocab = list(set(vocab))
    return vocab
