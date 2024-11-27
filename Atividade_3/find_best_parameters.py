import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def process_and_train(dataset_path):
    print(f"Processando o dataset: {dataset_path}")
    
    # Carregar e dividir os dados
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

    # Modelos
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=200, solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Treinar e avaliar
    for name, model in models.items():
        print(f"\nTreinando o modelo: {name}")
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_test_tfidf)
        print(f"Relatório para {name}")
        print(classification_report(y_test, y_pred))

# Processar ambos os datasets
datasets = [
    "C:/Users/tagsa/Downloads/classic4.csv",
    "C:/Users/tagsa/Downloads/Industry Sector.csv"
]

for dataset in datasets:
    process_and_train(dataset)

