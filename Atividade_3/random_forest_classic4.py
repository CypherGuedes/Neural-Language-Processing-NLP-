import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Caminho do dataset
dataset_path = "C:/Users/tagsa/Downloads/classic4.csv"

# Carregar e dividir o dataset
df = pd.read_csv(dataset_path)
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], df['class'], test_size=0.2, stratify=df['class'], random_state=42
)

# Pré-processamento
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

tfidf = TfidfTransformer()
x_train_tfidf = tfidf.fit_transform(x_train_vec)
x_test_tfidf = tfidf.transform(x_test_vec)

# Treinar e avaliar
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_tfidf, y_train)
predicted = model.predict(x_test_tfidf)

print("Relatório de Classificação - Random Forest (classic4.csv)")
print(classification_report(y_test, predicted))
