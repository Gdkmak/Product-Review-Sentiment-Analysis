from zipfile import ZipFile
import pandas as pd
import os 
import shutil

DATA = os.environ.get("TRAINING_DATA")

if __name__ == '__main__': 

  # extrat the zip file  
  with ZipFile('input/Turkish_Products_Sentiment.zip') as zip:
    zip.extractall('input/')

  os.remove('input/Turkish_Products_Sentiment.zip')

  lstdir = os.listdir('input/')
  lstdir = [(os.path.join('input', i,'positive') , 
            os.path.join('input', i,'negative')) for i in lstdir if not (i.startswith('.'))]

  # read the all files and add them together in a pandas dataframe
  df_lst = []
  for pos, neg in lstdir: 
    pos_data = pd.read_csv(pos, sep='\n', names=['sentiment'] , header=None)
    pos_data['target'] = 1
    neg_data = pd.read_csv(neg, sep='\n', names=['sentiment'] , header=None)
    neg_data['target'] = 0
    df_lst.extend([pos_data, neg_data])

  df = pd.concat(df_lst) 
  
  # shuffle data
  data = df.sample(frac=1)

  # take 10% of the data for test 
  split_perc = int(df.shape[0]*.10)

  test_data = data.iloc[0:split_perc ,:]
  train_data = data.iloc[split_perc: ,:]
  
  train_data.to_csv('input/train_data',index=None , header=None)
  test_data.to_csv('input/test_data',index=None , header=None)

  # delete unndeeded directories 
  dirs = ['dvd', 'electronics','kitchen','books']
  for dir in dirs:
    shutil.rmtree(os.path.join('input', dir))

