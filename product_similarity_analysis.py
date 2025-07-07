
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from nltk.stem import PorterStemmer


df = pd.read_csv("NikeProductDescriptions.csv")

# Preprocessing
stemmer = PorterStemmer()
stopwords = {
    'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'of', 'to', 'with', 'for',
    'by', 'is', 'it', 'this', 'that', 'as', 'from', 'be', 'are', 'was', 'were',
    'but', 'if', 'they', 'you', 'we', 'he', 'she', 'them', 'his', 'her', 'their',
    'its', 'our', 'your', 'not', 'so', 'do', 'does', 'did', 'out', 'can'
}

def preprocess(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}]", " ", text)
    tokens = re.findall(r'\b\w+\b', text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stopwords]
    return ' '.join(filtered)

df['Cleaned_Description'] = df['Product Description'].apply(preprocess)

#TF-IDF + Cosine Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Cleaned_Description'])
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

#  Jaccard Similarity
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    return len(set1 & set2) / len(set1 | set2)

results = []
for i, j in combinations(range(len(df)), 2):
    cosine_sim = cosine_sim_matrix[i, j]
    jaccard_sim = jaccard_similarity(df.loc[i, 'Cleaned_Description'],
                                     df.loc[j, 'Cleaned_Description'])
    results.append({
        'Product 1': df.loc[i, 'Title'],
        'Product 2': df.loc[j, 'Title'],
        'Cosine Similarity': round(cosine_sim, 3),
        'Jaccard Similarity': round(jaccard_sim, 3)
    })


similarity_df = pd.DataFrame(results)
top_similar = similarity_df.sort_values(by='Cosine Similarity', ascending=False).head(10)
print(" Top 10 Most Similar Product Descriptions ")
print(top_similar.to_string(index=False))

