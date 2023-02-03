import sys
import pandas as pd
from sqlalchemy import create_engine
# nlp tokenization
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
# pipeline setup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# nan replacing
import numpy as np
# save model
import pickle


'''
run as following to execute the program with correct arguments:
python models/train_classifier.py data/DisasterResponse_1.db models/classifier.pkl
'''
class ML:
    '''
    ML is the Machine-Learning helper class that creates the model object, using sklearn.pipeline
    - use this class to build (create), train (fit), evaluate, save, and load (read) the model 
    '''

    def __init__(self):
        '''
        initialize members, to use later for nltk processing
        '''
        self.stopwords_english = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()

    def db_to_df(self, db_name = 'data/DisasterResponse_1.db', tbl_name = 'MessagesCategoriesTable'):
        '''
        returns dataframe after reading sql from tablename
        
        input:
        - databasename: string, tablename: string
        output:
        - pandas dataframe
        '''
        engine = create_engine(f'sqlite:///{db_name}')
        df = pd.read_sql(f'SELECT * FROM {tbl_name}', engine)
        return df

    def load_data(self, database_filepath):
        '''
        loads X,Y i.e. the data, from database
        
        input:
        - database filepath
        output:
        - X,Y values and label strings
        '''
        df = self.db_to_df(db_name = database_filepath)
        X = df.loc[:,'message'].values
        Y = df.loc[:,'aid_centers':'weather_related']
        category_names = list(Y.columns)
        return (X, Y, category_names)

    def tokenize(self, text):
        '''
        returns tokens from input text

        input:
        - text string
        output:
        - list of tokens
        '''
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if not token in self.stopwords_english]
        tokens = [self.lemmatizer.lemmatize(t,pos='n') for t in tokens]
        tokens = [self.lemmatizer.lemmatize(t,pos='v') for t in tokens]
        return tokens

    def build_model_without_gridsearch(self):
        '''
        creates the model object, which can be used for prediction
        - pipeline is model, with type MultiOutputClassifier
        - this pipeline, as the name suggests, does not use gridsearch
        '''
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(LogisticRegression()))
        ])
        '''
        example to do both - fit model and predict, with build_model_without_gridsearch method:
        try:
            pipeline.fit(X_train, Y_train) # train classifier
            y_pred = pipeline.predict(X_test) # predict on test data
        except Exception as e: print(e)
        '''
        return pipeline

    def build_model(self):
        '''
        creates the model object, which can be used for prediction
        - pipeline is model, with type MultiOutputClassifier
        - this pipeline uses gridsearch
        '''
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(LogisticRegression()))
        ])
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': [False, True],
            'tfidf__smooth_idf': [False, True],
            'tfidf__sublinear_tf': [False, True],
        }
        cv = GridSearchCV(pipeline, param_grid=parameters)
        return cv

    def grid_results(self, model_obj):
        '''
        convenience method to print logs from model (object)
        '''
        print(" Results from Grid Search " )
        print("\n The best estimator across ALL searched params:\n",model_obj.best_estimator_)
        print("\n The best score across ALL searched params:\n",model_obj.best_score_)
        print("\n The best parameters across ALL searched params:\n",model_obj.best_params_)

    def evaluate_model(self, model, X_test, Y_test, category_names):
        '''
        convenience method to print model accuracy by matching `true` comparisons and returning the mean
        treating `false` matches as zero, and `true` matches as one, taking the `sum`, and dividing by the `count`
        '''
        Y_pred = model.predict(X_test)
        accuracy = (Y_pred == Y_test).mean()
        print(f'Model accuracy is {accuracy}')

    def save_model(self, model, model_filepath):
        '''
        save pipeline model at filepath using pickle
        '''
        with open(model_filepath, 'wb') as pkl_file:
            pickle.dump(model, pkl_file)
        return self.load_model(model_filepath)

    def load_model(self, model_filepath):
        '''
        load pipeline model from filepath using pickle
        '''
        model = None
        with open(model_filepath, 'rb') as pkl_file:
            model = pickle.load(pkl_file)
        return model

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        mlObj = ML()
        
        X, Y, category_names = mlObj.load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y.values, test_size=0.2)
        
        Y_train = np.where(np.isnan(Y_train), 0, Y_train)
        Y_test = np.where(np.isnan(Y_test), 0, Y_test)
        
        print('Building model...')
        model = mlObj.build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        mlObj.evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        success = mlObj.save_model(model, model_filepath)
        if success: print('Trained model saved!')
        else: print('Could not save trained model!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()