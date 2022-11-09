import sys
import nltk
nltk.download(['punkt', 'wordnet'])
               
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Loads the data from the database.

    Parameters:
    database_filepath: The path to the database file.

    :return: A list of tuples containing the data.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_details', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'original', 'genre', 'message']).values
    category_names = df.drop(columns=['id', 'original', 'genre', 'message']).columns
    return X,Y,category_names


def tokenize(text):
    """
    Tokenize the input text.

    Parameters:
    text: The input text.

    :return: List tokenized text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    """
    Builds the model.

    Returns:
    cv_model - a model combine with gridsearch and pipeline for vectorization and classification to find best specified parameters for predictions
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer= tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=25)))
    ])
    
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': (5 ,10)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates and display the model on the test data.

    Parameters:
    model: The training model.
    X_test: The test data.
    y_test: The test labels.
    category_names: The category names.
    """
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print(f"Category: {category_names[i]}")
        print(classification_report(y_test[:, i], y_pred[:, i]))
        print("\n")


def save_model(model, model_filepath):
    """
    Saves the model to the file system.

    Parameters:
    model: The model to be saved.
    model_filepath: The path to the model file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()