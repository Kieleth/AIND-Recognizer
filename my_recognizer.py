import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    sequences = test_set.get_all_sequences()

    for sequence in sequences:
        X_test, lengths_test = test_set.get_item_Xlengths(sequence)

        word_res = {}
        for word, model in models.items():
            try:
                logL = model.score(X_test, lengths_test)
            except Exception as e:
                #print('Exception for word %s' % (word))
                continue

            word_res[word] = logL

        if word_res:
            best_word = max(word_res.items(), key=lambda x: x[1])[0]
            probabilities.append(word_res)
            guesses.append(best_word)

    return probabilities, guesses
