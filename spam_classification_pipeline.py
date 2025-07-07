import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk

import warnings
warnings.filterwarnings("ignore")

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')

# === Chargement des données ===
df = pd.read_csv("spam.csv", encoding='latin-1')[['text', 'target']].dropna()

# === Prétraitement du texte ===
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = word_tokenize(text)
    return ' '.join([t for t in tokens if t not in stop_words and len(t) > 1])

df["clean"] = df["text"].apply(clean_text)

# === Vecteur TF-IDF pour clustering ===
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(df["clean"])

# === PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

# === Clustering ===
df["KMeans"] = KMeans(n_clusters=2, random_state=0).fit_predict(X_tfidf)
df["Agglo"] = AgglomerativeClustering(n_clusters=2).fit_predict(X_tfidf.toarray())
df["DBSCAN"] = DBSCAN(eps=1.2, min_samples=5).fit_predict(X_tfidf)

# === Affichage clustering ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["KMeans"])
plt.title("K-Means Clustering")
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["target"])
plt.title("Vraies étiquettes (spam/ham)")
plt.tight_layout()
plt.show()

# === Word2Vec ===
tokenized = df["clean"].apply(str.split)
w2v = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)

def vectorize(sentence):
    words = sentence.split()
    vecs = [w2v.wv[word] for word in words if word in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_w2v = np.vstack(df["clean"].apply(vectorize))
y = df["target"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(X_w2v, y, test_size=0.2, random_state=42)

# === Modèles supervisés ===
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": GridSearchCV(RandomForestClassifier(), {"n_estimators": [100, 200]}, cv=3),
    "Gradient Boosting": GridSearchCV(GradientBoostingClassifier(), {"n_estimators": [100, 150]}, cv=3),
    "XGBoost": GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss"), {"n_estimators": [100, 150]}, cv=3)
}

print("\n🎯 Résultats Classification Supervisée\n")

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=["ham", "spam"])
        print(f"--- {name} ---")
        print(f"Accuracy : {acc:.3f}")
        print("Classification Report:\n", report)
    except Exception as e:
        print(f"{name} - Erreur : {str(e)}")

print("\n✅ Script terminé.")
