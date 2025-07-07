
import nltk
import pandas as pd
from nltk.corpus import brown
from collections import Counter
import re


nltk.download('brown')
tokenized_text = [word.lower() for word in brown.words() if re.match(r'\w+', word)]


def get_ngrams(tokenized_text, n):
    ngrams = zip(*[tokenized_text[i:] for i in range(n)])
    ngram_freq = Counter(ngrams)
    df = pd.DataFrame(ngram_freq.items(), columns=['ngram', 'frequency']).sort_values(by='frequency', ascending=False)
    return df


def predict_next_words(context, n, k, ngram_df):
    context = tuple(context.lower().split()[-(n-1):])  # get last (n-1) words
    matches = ngram_df[ngram_df['ngram'].apply(lambda x: x[:-1] == context)]
    top_k = matches.sort_values(by='frequency', ascending=False).head(k)
    return top_k[['ngram', 'frequency']]

# === 4. Example Usage ===
if __name__ == "__main__":
    n = 3  # Trigram model
    k = 5  # Top 5 suggestions
    ngram_df = get_ngrams(tokenized_text, n)
    print("\nTop 5 predictions for input 'the united':")
    print(predict_next_words("the united", n, k, ngram_df))
