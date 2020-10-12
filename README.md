# Illuin_test
Document retrieval task

Arguments :

-d Datasets available : fquad, squad (v2.0) and squad1 (v1.1)  
-sp Split (train, valid, test, merge) : use the training, validation or test set. The merge option creates a new dataset with the contexts
      from the training and validation set, and evaluates the algorithm on the validation questions. This makes
      the retrieval task harder as there are more contexts to search in.  
-s Sample : Number of questions to evaluate the algorithm on. It is useful to evaluate the algorithm on a subset of questions as the runtime on the whole dataset may
       be quite important  
-e Whether to evaluate the algorithm on a subset of questions, or to predict the context for a given question id  
-id Index of the question to retrieve the context for (case evaluate=False)  
-k Number of contexts that the algorithm will retrieve for a given question  
       
       
## Use cases
Prediction on a given question : python main.py -d fquad -sp valid -id 100 -k 20  
Evaluation on a subset of questions :               python main.py -d squad1 -sp merge -e -s 2500  

On my machine : python main.py -d fquad -e -sp valid -s 3188  
runs in 17.29s and yields : Accuracy: 0.517, Accuracy@5: 0.738, Accuracy@20: 0.856
