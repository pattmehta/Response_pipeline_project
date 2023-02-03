import sys
import pandas as pd
import re
from sqlalchemy import create_engine

'''
run as following to execute the program with correct arguments:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse_1.db
'''
class ETL:
    '''
    ETL is the Extract-Transform-Load helper class that
    - extracts data from multiple filepaths
    - transforms the data, into a usable format
    - loads (puts) the data into a database
    '''

    def __init__(self):
        '''
        initialize members, to use later for transforming the data
        '''
        self.category_colnames = None
        self.categories_csv_col = None

    def get_category_column_names(self, categories):
        '''
        provides the category labels, and assigns it to the property `category_colnames`
        '''
        if self.category_colnames is not None: return self.category_colnames

        catre = re.compile(r'-[0-9]')
        # select the first row of the categories dataframe
        categories_rowlist = categories[0].split(';')
        # create category names
        self.category_colnames = list(map(lambda x: re.sub(catre,'',x),categories_rowlist))
        return self.category_colnames

    def get_category_list(self, rowlist):
        '''
        transformer that converts `related-1;request-1;offer-0;` to `['related','request']` to `'related|request'`
        - retains labels with a value of 1
        - the "|" character join, is an important step
        - a join of string labels, with "|", is used by the pandas `get_dummies` method to create an appropriate dataframe with auto-generated columns
        - this is a kind of one-hot-encoding
        - this method is a transformer, i.e., it is applied in a row-wise manner
        '''
        colnames = self.get_category_column_names(self.categories_csv_col)
        codelist = []
        for i,x in enumerate(rowlist):
            value = x.split('-')[1]
            value = int(value)
            if value == 1: codelist.append(colnames[i])
        return '|'.join(codelist)

    def categories_to_df(self, categories):
        '''
        transforms each row to a string of label names (with a value of 1), joined by "|" character
        
        input:
        - categories (a pandas series column)
        output:
        - dataframe, where columns are all the labels, and values are one-hot-encoded (1, or 0, based on the label existence)
        '''
        categories_eq_one_series = categories.apply(lambda col: self.get_category_list(col.split(';')))
        # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.get_dummies.html#pandas.Series.str.get_dummies
        # default string to split on is the "|" character
        categories_encoded = categories_eq_one_series.str.get_dummies() # categories are sorted by alphabet [aid_centers, aid_related, buildings, ...]
        return categories_encoded

    def load_data(self, messages_filepath, categories_filepath):
        '''
        facade method that calls various useful methods for ETL
        
        input:
        - filepaths to read the data
        output:
        - concatenated dataframe
        '''
        messages = pd.read_csv(messages_filepath)
        messages.drop_duplicates(inplace=True,subset=['id'],keep=False)

        categories = pd.read_csv(categories_filepath)
        categories.drop_duplicates(inplace=True,subset=['id'],keep=False)

        self.categories_csv_col = categories['categories']
        df = categories.merge(messages,on='id')
        categories_encoded = self.categories_to_df(self.categories_csv_col)
        return pd.concat([df.reset_index(drop=True),categories_encoded.reset_index(drop=True)],axis=1)

    def clean_data(self, df):
        '''
        additional cleaning step
        '''
        df.drop('categories',axis=1,inplace=True)
        df.dropna(axis=0,subset=['id'],inplace=True)
        df['id'] = df['id'].astype('int32')
        return df.drop_duplicates()

    def save_data(self, df, database_filename):
        '''
        saves the data to a database
        
        input:
        - dataframe, and database filename for saving data
        output:
        - boolean indicating success or failure
        '''
        success = True
        engine = create_engine(f'sqlite:///{database_filename}')
        try: df.to_sql('MessagesCategoriesTable', engine, index=False)
        except Exception as e: success = False
        # to read
        # sqldf = pd.read_sql("SELECT * FROM MessagesCategoriesTable", engine)
        return success

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        etlObj = ETL()

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = etlObj.load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = etlObj.clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        success = etlObj.save_data(df, database_filepath)
        
        if success: print('Cleaned data saved to database!')
        else: print('Could not save to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()