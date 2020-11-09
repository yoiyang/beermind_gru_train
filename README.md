## Files

configs_beerstyle.py: numeric mapping of beer styles

configs.py: configurations of the model

functions.py: functions needed in main_ files

function_oh_rat.py: functions that implements one hot encoding of the rating

main_cont.py: continues to train a half-finished model

main_generate.py: generates reviews given the test file

main_no_dispatcher.py: trains a model

main_test_bleu_same.py: using one-to-one calculation for bleu score (see report)

main_test_bleu.py: using many-to-one calculation to find bleu score

main_test_original_bleu.py: using one-to-one calculation to find bleu score given the generated reviews

models.py: has class definitions of GRU and LSTM

## Get started

0. Check your environment. Requirement: python 3, pytorch, cuda, BeerAdvocatesPA4 datasets.
1. Modify configs.py with ideal hyperparameters before training.
2. Use main_no_dispatcher.py to train a model.
3. (If there is a interrpt) Use main_cont.py to continue generate; remember to input a saved model path.
4. Use main_generate.py to see generate texts. Later use main_test_original_bleu.py to test with the file generated.
4. Use main_test_bleu_same.py to test its bleu score given the test data.
