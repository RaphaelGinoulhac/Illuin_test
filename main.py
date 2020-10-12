from utils import *
import random
import argparse
import time
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="fquad", choices=["fquad", "squad", "squad1"],
                        help="Options : fquad, squad, squad1")
    parser.add_argument("-sp", "--split", default="train", choices=["train", "valid", "test", "merge"],
                        help="Options : train, valid, test, merge")
    parser.add_argument("-s", "--sample", default=500, type=int,
                        help="Number of questions to evaluate the algorithm on, sampled randomly (case evaluate=True)")
    parser.add_argument("-e", "--evaluate", default=False, action='store_true',
                        help="Whether to evaluate the algorithm on a subset of questions, or to predict the context"
                             "for a given question id")
    parser.add_argument("-id", "--index", default=0, type=int,
                        help="Index of the question to retrieve the context for (case evaluate=False)")
    parser.add_argument("-k", "--k", default=20, type=int,
                        help="Number of contexts that the algorithm will retrieve for a given question")
    args = parser.parse_args()

    # Datasets available : fquad, squad (v2.0) and squad1 (v1.1)
    # Split : use the training, validation or test set. The merge option creates a new dataset with the contexts
    #         from the training and validation set, and evaluates the algorithm on the validation questions. This makes
    #         the retrieval task harder as there are more contexts to search in.
    # Sample : It is useful to evaluate the algorithm on a subset of questions as the runtime on the whole dataset may
    #          be quite important
    # Use cases : python main.py -d fquad -sp valid -id 100 -k 20
    #            python main.py -d squad1 -sp merge -e -s 2500

    # On my machine : python main.py -d fquad -e -sp valid -s 3188
    # runs in 17.29s and yields : Accuracy: 0.517, Accuracy@5: 0.738, Accuracy@20: 0.856
    data_dir = "data/"

    if args.split != "merge":
        print("Loading the dataset")
        data = load_json(args.dataset, args.split, data_dir)
        print("Creating a context-question dataset")
        dataset, contexts = create_dataset(data)
        questions = [elt[1] for elt in dataset]
    else:
        # merge the train and validation contexts to search in, and evaluate on the validation questions
        print("Loading the dataset")
        data_train = load_json(args.dataset, "train")
        data_dev = load_json(args.dataset, "valid")
        print("Creating a context-question dataset")
        _, contexts_train = create_dataset(data_train)
        dataset, contexts_dev = create_dataset(data_dev)
        questions = [elt[1] for elt in dataset]
        contexts = contexts_train + contexts_dev
    print("There are {} questions and {} unique contexts in this dataset".format(len(questions), len(contexts)))

    idf_filename = data_dir + args.split + "_" + args.dataset + ".p"
    if not os.path.exists(idf_filename):
        print("Computing an idf table")
        idf = build_idf(contexts)
        pickle.dump(idf, open(idf_filename, "wb"))
    else:
        idf = pickle.load(open(idf_filename, "rb"))

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
