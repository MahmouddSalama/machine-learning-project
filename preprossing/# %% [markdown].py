# %% [markdown]
# impor lips

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# read data

# %%
dataframe=pd.read_csv("player-value-prediction.csv")
print(dataframe.shape)
dataframe.head(3)

# %% [markdown]
# data info

# %%
dataframe.info()

# %% [markdown]
# data describe

# %%
dataframe.describe()

# %% [markdown]
# data columns

# %%
dataframe.columns

# %% [markdown]
# delete un use columns

# %%
# drop nulll & unusfull columns 
nullcols=['id', 'name', 'full_name', 'birth_date','traits','national_team','national_rating','national_team_position','national_jersey_number','club_join_date','contract_end_year','tags','traits']
dataframe.drop(nullcols,axis=1,inplace=True)
dataframe.shape

# %% [markdown]
# sum of null

# %%
dataframe.isna().sum()

# %% [markdown]
# know how max num of postions can player pkay on it 

# %%
c= dataframe['positions'].str.split(',').max()
c # max num of position can player play on it in data is 

# %% [markdown]
# split position data

# %%
def spliPostion(data):
    max_values = np.max([
        data['positions'].apply(lambda x: len(str(x).split(','))).max(),
    ])
    return data['positions'].str.split(',').apply(lambda x: x + ['0'] * (max_values - len(x)))

# %%
# the lare number player can play in it is 3
play_postion = spliPostion(dataframe)
dataframe['position_first'] = play_postion.apply(lambda x: x[0])
dataframe['position_second'] = play_postion.apply(lambda x: x[1])
dataframe['position_third'] = play_postion.apply(lambda x: x[2])
dataframe.drop('positions', axis=1, inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder
transform=['nationality','preferred_foot','body_type','club_team''club_position','position_first','position_second','position_third']

le=LabelEncoder()
dataframe['nationality']=le.fit_transform(dataframe['nationality'])
dataframe['nationality']=le.fit_transform(dataframe['nationality'])
dataframe['preferred_foot']=le.fit_transform(dataframe['preferred_foot'])
dataframe['body_type']=le.fit_transform(dataframe['body_type'])
dataframe['club_team']=le.fit_transform(dataframe['club_team'])
dataframe['club_position']=le.fit_transform(dataframe['club_position'])

dataframe['position_first']=le.fit_transform(dataframe['position_first'])
dataframe['position_second']=le.fit_transform(dataframe['position_second'])
dataframe['position_third']=le.fit_transform(dataframe['position_third'])

# %%
dataframe['upper_work_rate'] = dataframe['work_rate'].apply(lambda x: x.split('/')[0])
dataframe['down_work_rate'] = dataframe['work_rate'].apply(lambda x: x.split('/')[1])


# %%
dataframe['height_cm'] = dataframe['height_cm'].apply(lambda x:x/100)

# %%
dataframe.head(3)

# %%
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
dataframe['release_clause_euro']=minmax.fit_transform(dataframe[['release_clause_euro']])
dataframe['wage']=minmax.fit_transform(dataframe[['wage']])

# %%
def colHasAddToNum(data, col,):
    def multi(values):
        result = None
        values = [float(val) for val in values]
        if values is None or np.isnan(values).any():
            result = np.nan
        elif len(values) == 1:
            result = float(values[0])
        elif len(values) == 2:
            result = float(values[0]) + int(values[1])
        else:
            result = values
        return result
    
    return data[col].apply(lambda x: multi(str(x).split('+')))

# %%


# %%
exp_cols = []
for col in dataframe.columns:
    if dataframe[col].apply(lambda x: str(x).split('+')).dropna().apply(lambda x: len(x)).max() > 1:
        exp_cols.append(col)
exp_cols

# %%
for col in exp_cols:
    dataframe[col] = colHasAddToNum(dataframe, col)
    

# %%
dataframe[exp_cols].head()

# %%
dataframe.isna().sum()

# %%
def convert_rate(col):
    dataframe[col]=le.fit_transform(dataframe[col])
    dataframe[col]=dataframe[col].apply(lambda x:x+1)

convert_rate('upper_work_rate')
convert_rate('down_work_rate')

# %%
dataframe['num_of_work_rate']=dataframe['upper_work_rate']/dataframe['down_work_rate']
dataframe['num_of_work_rate'].head()

# %%
dataframe= dataframe.drop(['work_rate'],axis=1)

# %%
for col in dataframe.columns:
    if dataframe[col].isnull:
        
        dataframe[col].fillna(dataframe[col].mean(),inplace=True)

# %%


# %%
dataframe.head()

# %% [markdown]
# ## outlayers  

# %%
cleandata=pd.read_csv("clean_data2.csv")
cleandata.shape
cleandata.head()

# %%
from matplotlib import pyplot as plt
from scipy.stats import norm
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10,6)
plt.hist(cleandata['value'],bins=50,rwidth=.8)

plt.xlabel('value (inches)')
plt.ylabel('Count')
plt.show()

# %%
Q1=cleandata['value'].quantile(.25)
Q2=cleandata['value'].quantile(.50)
Q3=cleandata['value'].quantile(.75)
Q1,Q2,Q3

# %%
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
lower,upper

# %%
cleandata['value'].std(),cleandata['value'].mean()

# %%
minlarer=cleandata['value'].mean()-5687164*cleandata['value'].std()
maxlayer=cleandata['value'].mean()+5687164*cleandata['value'].std()
minlarer,maxlayer

# %%
cleandata = cleandata[cleandata['value']>lower]

# %%
cleandata = cleandata[cleandata['value']<5262500]

# %%
cleandata.head()

# %%
plt.hist(cleandata['value'],bins=50,rwidth=.8)

plt.xlabel('value (inches)')
plt.ylabel('Count')
plt.show()

# %%


# %%
sh={
    'dataframe':{
        'Y':dataframe.shape[0],
        'X':dataframe.shape[1]
    },
    'cleanDate':{
        'Y':cleandata.shape[0],
        'X':cleandata.shape[1]
    },
}

out=pd.DataFrame(sh)

# %%
out

# %% [markdown]
# 

# %%
'''
1- import helper lips
2- read data fromcsv
3- print data info
4- print data describe
5- print data columns names
6- drop nulll & unusfull columns 
7- print fraction of null valuses in columns
8- split column position 
9- give new columns of pposition values
10- lable Encoder form uon numecic data
11- split work rate in to  upper_work_rate & down_work_rate
12- scaling large values ( MinMaxScaler )
13- columns of ( + ) split and sum values to gether
14- lable Encoder upper_work_rate & down_work_rate
15- devide 2 columns
16- fill null values of mean values
17- remove the outlayer
18- get csv file from dataframe (clean_data1.csv)
'''

# %%
dataframe.shape

# %%


# %%



