from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar e processar os dados
df = pd.read_csv("C:/Users/tagsa/Downloads/classic4.csv")
x_train, x_test = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)

# Vetorização e transformação TF-IDF
vectorizer = CountVectorizer()
x_train_tf = vectorizer.fit_transform(x_train['text'])
x_test_tf = vectorizer.transform(x_test['text'])

tfidf = TfidfTransformer()
x_train_tfidf = tfidf.fit_transform(x_train_tf)
x_test_tfidf = tfidf.transform(x_test_tf)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_tfidf, x_train['class'])

# Prever e avaliar
predicted = model.predict(x_test_tfidf)
print(classification_report(x_test['class'], predicted))
