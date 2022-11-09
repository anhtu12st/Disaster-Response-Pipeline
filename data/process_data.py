import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load draft data from csv file and merge base on 'id' column

    Parameters:
    messages_filepath (str): messages csv file path
    categories_filepath (str): categories csv file path

    Returns:
    DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    print("Data loaded: ", df.head())
    return df


def clean_data(df):
    """
    Clean and transform DataFrame to categorical columns
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Because of some columns in the data contain more than 2 unique values
    # We need to transform other values that different from "0" to be one
    categories = categories.applymap(lambda x: 0 if x == 0 else 1)
    
    df.drop(columns=['categories'], inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop_duplicates()
    print("Data cleaned: ", df.head())
    return df


def save_data(df, database_filename):
    """
    Save DataFrame to db

    Parameters:
    df (DataFrame): DataFrame to save
    database_filename (str): database filename
    """
    df.to_csv(database_filename, index=False)
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_details', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()