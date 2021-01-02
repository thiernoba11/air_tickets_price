#!/usr/bin/env python
# coding: utf-8

# In[6]:


import json
import pandas as pd
import os
import re
from tqdm.notebook import tqdm


# In[7]:


def get_filepaths(rootdir='.', filetypes=['txt']):
    
    """
    Returns the path-string of all the json files in the given root dirctory.

    Usage:
        from utils import get_filepaths

        get_filepaths(rootdir='~/path/to/dir', filetypes=['txt'])

    Parameters:
        rootdir:
              The directory where the files are read from.
              The default is '.'

        filetypes:
              A list containing the string form of the file types to read.
              Default: ['txt']  reads all text files
              pass an empty list [] for reading all file types

    This function raises an exception if the given path does not exist in the local system.
    """
    # If found, remove the '/' at the end of rootdir 
    if rootdir[-1] == os.sep:
        rootdir = rootdir[:-1]

    # If the directory does not exist on the user system, then raise an exception error
    if os.path.exists(rootdir) is False:
        raise Exception(f"Directory `{rootdir}` does not exist.")

    # Go through the folder structure and add to filepaths list
    filepaths = []
    # Convert filetypes to lower case
    filetypes = [ftype.lower() for ftype in filetypes]

    for (dirpath, dirnames, filenames) in os.walk(rootdir):
        for filename in filenames:
            # if filename is in given filetypes
            if filetypes == []:
                filepaths.append(os.path.join(dirpath, filename))
            else:
                # Split the filename with . and check if it is the desired extension
                if filename.split('.')[-1].lower() in filetypes:
                    filepaths.append(os.path.join(dirpath, filename))

    # return the filepaths list
    return filepaths


# In[8]:


filepaths = get_filepaths('./dataset1')
# filepaths.extend(get_filepaths('./dataset2'))


# In[9]:


all_data = []


# In[10]:


for fp in tqdm(filepaths):
    data = json.load(open(fp, 'r'), encoding = 'UTF-8')
    all_data.extend(data)    


# In[11]:


JSON_df = pd.DataFrame(all_data)
JSON_df.head()


# In[ ]:


# Expand nested data from Flights column and drop rows with NA values
df = JSON_df.explode(column='Flights').dropna()
df.reset_index(inplace=True, drop=True)


# In[13]:


# Delete Flights column and concatenate the newly generated columns to the main dataframe
df = pd.concat([df.drop(['Flights'], axis=1), df['Flights'].apply(pd.Series)], axis=1)
df.head()


# In[14]:


# df.to_csv('airtickets.csv')


# In[15]:


# df.info()


# In[16]:


df['CurrentDate'] = pd.to_datetime(df['CurrentDate'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['STD'] = pd.to_datetime(df['STD'], errors='coerce')
df['STA'] = pd.to_datetime(df['STA'], errors='coerce')


# In[17]:


df['HasSelection'] = df['HasSelection'].astype('bool')
df['InMonth'] = df['InMonth'].astype('bool')
df['IsMACstation'] = df['IsMACStation'].astype('bool')
df['IsAirportChange'] = df['IsAirportChange'].astype('bool')


# In[18]:


df['FlightNumber'] = df['FlightNumber'].astype(str)
df['ArrivalStationCode'] = df['ArrivalStationCode'].astype(str)
df['DepartureStationCode'] = df['DepartureStationCode'].astype(str)
df['ArrivalStationName'] = df['ArrivalStationName'].astype(str)
df['DepartureStationName'] = df['DepartureStationName'].astype(str)


# In[19]:


df['MinimumPrice'] = df['MinimumPrice'].astype('str')


# In[20]:


# df.info()


# In[21]:


# df['MinimumPrice'].value_counts()


# In[22]:


def clean_currency(x):
    """ Remove comma in currency numeric
        Add space between currency symbol and value
    """
    if isinstance(x, int) or isinstance(x, float):
        return ('', x)
    if isinstance(x, str):
        numeric_start_at = 0
        for i in range(len(x)):
            if x[i].isnumeric():
                numeric_start_at = i
                break
        return (x[0:numeric_start_at], x[numeric_start_at:].replace(',', ''))
    return('', x)


# In[23]:


df['MinimumPrice'] = df['MinimumPrice'].apply(clean_currency).astype('str')


# In[24]:


df1 = df.head(1000).apply(lambda x: pd.Series(clean_currency(x['MinimumPrice']),
                                              index=['symbol', 'value']), axis=1, result_type='expand')


# In[25]:


currency_correction_map = {
    "â‚¬": "EUR",
    "lei": "LEI",
    "Ft" : "FT",
    "Kr" : "KR",
    "MKD" : "MKD"
}

def currency_correction(x):
    if x in currency_correction_map:
        return currency_correction_map[x]
    else:
        return x


# In[26]:


df1['symbol'] = df1['symbol'].apply(currency_correction)
df1


# In[27]:


currency_multiplier_EUR = {
    "EUR": 1,
    "LEI": 0.21,
    "FT" : 0.0027,
    "KR" : 0.099,
    "MKD" : 0.016       
}

def currency_converter(symbol, value):
    value = float(value)
    if symbol in currency_multiplier_EUR:
        return currency_multiplier_EUR[symbol] * value
    else:
        return None


# In[28]:


df1['Price_in_EUR'] = df1.apply(lambda x: currency_converter(x['symbol'], x['value']), axis=1)
df1


# In[ ]:


df2 = df.head(1000)
final_df = pd.concat([df2.drop(['MinimumPrice'], axis=1), df1], axis=1)


# In[ ]:




