from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Carregar os dados
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dados carregados de {filepath}")
    return df

# Pré-processar os dados
def preprocess_data(df):
    x_train, x_test = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)
    count_vectorizer = CountVectorizer()
    x_train_tf = count_vectorizer.fit_transform(x_train['text'])
    x_test_tf = count_vectorizer.transform(x_test['text'])
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_tf)
    x_test_tfidf = tfidf_transformer.transform(x_test_tf)
    return x_train_tfidf, x_test_tfidf, x_train['class'], x_test['class']

# Realizar GridSearchCV
def grid_search_model(model, param_grid, x_train, y_train, model_name):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(x_train, y_train)
    print(f"{model_name} Grid Search")
    print(f"Melhor score: {grid_search.best_score_:.3f}")
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    return grid_search

# Salvar resultados
def save_results(results, output_path):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Resultados salvos em {output_path}")

# Principal
if __name__ == "__main__":
    filepath = "C:/Users/tagsa/Downloads/classic4.csv"
    df = load_data(filepath)
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Modelos e parâmetros
    models = [
        ("Naive Bayes", MultinomialNB(), {'alpha': [0.1, 0.5, 1.0], 'fit_prior': [True, False]}),
        ("Logistic Regression", LogisticRegression(), {'C': [0.1, 1.0, 10], 'solver': ['liblinear', 'saga']}),
        ("Random Forest", RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [None, 10]})
    ]

    results = []
    for model_name, model, param_grid in models:
        grid = grid_search_model(model, param_grid, x_train, y_train, model_name)
        best_model = grid.best_estimator_
        test_score = best_model.score(x_test, y_test)
        results.append({
            "Model": model_name,
            "Best Score": grid.best_score_,
            "Best Parameters": grid.best_params_,
            "Test Score": test_score
        })

    # Salvar os resultados
    save_results(results, "C:/Users/tagsa/Downloads/Grid_Search_Results.csv")
