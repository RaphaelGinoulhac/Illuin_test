import random
from utils import *
import argparse
import time
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="train",
                        help="Options : train, valid, test, train_squad, dev_squad, train_squad1, dev_squad1")
    parser.add_argument("-s", "--sample", default=500, type=int,
                        help="Number of questions to evaluate the algorithm on, sampled randomly")
    parser.add_argument("-e", "--evaluate", default=False, action='store_true',
                        help="Whether to evaluate the algorithm on a subset of questions, or to predict the context"
                             "for a given question id")
    parser.add_argument("-id", "--index", default=0, type=int,
                        help="Index of the question to retrieve the context for")
    parser.add_argument("-k", "--k", default=10, type=int,
                        help="Number of contexts that the algorithm will retrieve for a given question")
    args = parser.parse_args()

    if args.dataset != "merge1":
        print("Loading the dataset")
        data = load_json(args.dataset + ".json")
        print("Creating a context-question dataset")
        dataset, contexts = create_dataset(data)
        questions = [elt[1] for elt in dataset]
    else:
        # merge the train and dev contexts to search in, and evaluate on the dev questions
        print("Loading the dataset")
        data_train = load_json("train_squad1.json")
        data_dev = load_json("dev_squad1.json")
        print("Creating a context-question dataset")
        _, contexts_train = create_dataset(data_train)
        dataset, contexts_dev = create_dataset(data_dev)
        questions = [elt[1] for elt in dataset]
        contexts = contexts_train + contexts_dev
    print("There are {} questions and {} unique contexts in this dataset".format(len(questions), len(contexts)))

    # fr_embeddings_path = download_word2vec()

    if not os.path.exists(args.dataset + ".p"):
        # word2vec = Word2Vec(fr_embeddings_path, vocab_size=250000)
        # sentence2vec = BagOfWords(word2vec, contexts)

        print("Computing an idf table")
        # idf = sentence2vec.build_idf(contexts)
        idf = build_idf(contexts)
        pickle.dump(idf, open(args.dataset + ".p", "wb"))

    else:
        idf = pickle.load(open(args.dataset + ".p", "rb"))
    # sentence2vec.encode_sentences(idf)
    start = time.time()

    if args.evaluate:
        sample = min(args.sample, len(questions))
        subset = random.sample(range(len(questions)), sample)
        print(f"Evaluating the algorithm on {sample} questions")
        predictions = naive_predict(questions, contexts, idf, subset, method="unweighted", k=args.k)
        acc, acc_five, acc_k = evaluate_retrieval(dataset, predictions, subset)
        print(f"Accuracy: {acc:.3f}, Accuracy@{min(5, args.k)}: {acc_five:.3f}, Accuracy@{args.k}: {acc_k:.3f}")

    else:
        idx = min(args.index, len(questions))
        question = questions[idx]
        print("Question: ", question)
        similar_sentences = naive_search(question, contexts, idf, method="unweighted", k=args.k)

        for i, sentence in enumerate(similar_sentences):
            print(str(i + 1) + ')', sentence)

        print("\nTrue context: ", dataset[idx][0])
    print(f"Prediction step executed in {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
