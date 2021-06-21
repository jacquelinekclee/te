def tfidf_data(review, reviews):
    all_words = review.split()
    words = set(all_words)
    cnt = list(map(lambda word: all_words.count(word), words))
    df = pd.DataFrame(data = {'cnt':cnt}, index = words)
    num_words = len(all_words)
    documents = reviews.str.split()
    num_docs = documents.size
    df['tf'] = df['cnt'] / num_words
    df['idf'] = list(map(lambda word: np.log(num_docs / documents.apply(lambda lst: (lst.count(word))>0).sum()), words))
    df['tfidf'] = df['tf'] * df['idf']
    return df


def relevant_word(out):
    return out['tfidf'].idxmax()


# The `train` method takes in a list of tokens (e.g. the output of `tokenize`) and outputs a language model. This language model is usually represented as a `Series` (indexed by tokens; values are probabilities that token occurs), or a `DataFrame`.
# The `probability` method takes in a sequence of tokens and returns the probability that this sequence occurs under the language model.
# *he `sample` method takes in a number `N > 0` and generates a string made up of `N` tokens using the language model. This method generates language.

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams

        self.mdl = self.train(ngrams)


        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

        


    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.
        """
        return [tuple(tokens[i:i+self.N]) for i in range(0, len(tokens)-self.N+1)]
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).
        """
        ngrams_counts = pd.Series(ngrams).value_counts().to_frame(name = 'ngram counts').reset_index()
        ngrams_counts['n-1 grams'] = ngrams_counts['index'].apply(lambda gram: gram[:-1])
        n1grams = pd.Series(ngrams).apply(lambda gram: gram[:-1])
        n1grams_counts = pd.Series(n1grams).value_counts().to_frame(name = 'n-1 gram counts')
        ngrams_counts.index = ngrams_counts['n-1 grams']
        ngrams_counts = ngrams_counts.drop(columns=['n-1 grams'])
        merged = n1grams_counts.merge(ngrams_counts, left_index=True, right_index=True)
        merged = merged.rename(columns={'index':'ngram'})
        merged = merged.reset_index()
        merged = merged.rename(columns={'index':'n1gram'})
        merged['prob'] = merged['ngram counts'] / merged['n-1 gram counts']
        return merged.drop(columns=['ngram counts','n-1 gram counts'])[['ngram', 'n1gram', 'prob']]
    
    def generate_models_dict(self):
        if isinstance(self, UnigramLM):
            return {1:self.mdl}
        current = self
        # create dictionary storing all n through 1 models
        models = {self.N:self.mdl}
        # loop until reach the UnigramLM (where n=1)
        while not isinstance(current, UnigramLM):
            if isinstance(current, UnigramLM):
                break
            # if an NGramLM, use its N to store in dict
            else:
                if isinstance(current.prev_mdl, NGramLM):
                    models[current.prev_mdl.N] = current.prev_mdl.mdl
                # if you've reached the UnigramLM, use 1
                else:
                    models[1] = current.prev_mdl.mdl
                # update current
                current = current.prev_mdl

        return models

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.
        """
        models = self.generate_models_dict()
        
        prob = 1
        
        # n tells us the maximum number of words in the denominator/n1gram
        n = self.N - 1

        # word doesnt exist

        # loop through all the words
        for i, word in enumerate(words):
            # for the first word, there are no conditional prob
            # access the unigram
            if i == 0:
                if word not in list(models[1].index):
                    return 0
                else:
                    prob *= models[1][word]
            else:
                # access the previous n words 
                # if previous n words exceeds bounds of tup,
                # go to the first element of the tup
                n1gram = tuple(words[i-min(i, n):i])
                # add the current word to the above n1gram
                ngram = tuple(words[i-min(i, n):i+1]) 
                # ngram is the numerator; use its length
                # to determine which model (which N) to use
                model = models[len(ngram)]
                # find the probability associated with the 
                # ngram and n1gram given the appropriate model
                df = model.loc[(model['ngram'] == ngram) & (model['n1gram'] == n1gram)]
                if df['prob'].size == 0:
                    return 0
                else:
                    prob *= df['prob'].iloc[0]
        return prob

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.
        """
        # start with \x02
        string = ['\x02']

        # get dict of all 2-n models
        models = self.generate_models_dict()
        # Use a helper function to generate sample tokens of length `length`
        def one_word_sample(ngram):
            # input is n-1 gram; get mdl corresponding to it
            model = models[len(ngram)+1]
            # get entries where n1gram equals input
            potential = model.loc[model['n1gram'] == ngram]
            # if no corresponding data, start over with STOP token
            if potential.shape[0] == 0:
                return ('\x03',)
            else:
                probs = potential['prob']
                ngram_samp = np.random.choice(a=potential['ngram'], p=probs)
                return ngram_samp
        # curr stores most recent sampled ngram, start with \x02
        curr = ('\x02',)
        while (len(string)-1) < M:
            # generate sample based on conditional probability
            to_add = one_word_sample(curr)
            # only add end of ngram returned
            string.append(to_add[-1])
            # update most recent sample ngram
            if len(string) >= self.N:
                # make sure only use n-1 grams
                curr = tuple(string[-1 * (self.N-1):])
            else:
                curr = tuple(string)
            # break when done
            if (len(string)-1) == M:
                break

        # Transform the tokens to strings
        return ' '.join(string)