import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

#create log directory, if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#create logging object
logger = logging.getLogger('data_ingestion')
#set logger level
logger.setLevel('DEBUG')

#create log console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#create log file handler
log_file_path = os.path.join(log_dir,'data_ingestion_log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#specify log formatter
#asctime: provides the loging time in a readable format, while 's' denotes that we want to convert the time into string format
#name: denotes the logfile name, in this case -> data_ingestion
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#set the above format for both console and file handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#after defining the log handlers, add them back to logger object belonging to logging class
logger.addHandler(console_handler)
logger.addHandler(file_handler)

#function to load data (note: pd.Dataframe after '->' denotes return type hint/suggestion)
def load_data(data_url:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded for %s', data_url)
        return df
    #exception handling to capture parsing error while loading data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse csv data file from: %s', e)
        raise
    #handle other errors
    except Exception as e:
        logger.error('Unexpected error while trying to load data from: %s', e)
        raise

#function to preprocess data
def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

#function to save train and test data after splitting the main dataframe
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        #create directory to save train and test data files
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        #save data files to above path
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

#define main function:
def main():
    try:
        #specify test dataframe size
        test_size = 0.2
        #specify data file url for ingestion
        data_path = 'https://raw.githubusercontent.com/Arko2016/Datasets/refs/heads/master/spam.csv'
        #invoke load_data function to ingest data from above url
        df = load_data(data_url=data_path)
        #process data
        processed_df = preprocess_data(df)
        #split to train and test
        #random state helps to randomly sample the data between train and test
        train_df, test_df = train_test_split(processed_df, test_size=test_size, random_state=3)
        #save train and test data in specified path
        save_data(train_data = train_df, test_data=test_df, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete data ingestion process : %s', e)
        raise

#the below format is a Python construct which allows the code to be executed only when the file is run as a script and not when its imported as a module by another script
if __name__ == '__main__':
    main()







