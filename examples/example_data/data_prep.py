# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# df1=pd.read_csv('RedBluff_1.csv')
# df2=pd.read_csv('RedBluff_2.csv')
# df3=pd.read_csv('RedBluff_3.csv')
# df4=pd.read_csv('RedBluff_4.csv')
# df5=pd.read_csv('RedBluff_5.csv')
# df6=pd.read_csv('RedBluff_6.csv')


df1=pd.read_csv('SF_1.csv')
df2=pd.read_csv('SF_2.csv')
df3=pd.read_csv('SF_3.csv')
df4=pd.read_csv('SF_4.csv')


# df1[['Month','Day','Year']] = df1['DATE'].str.split('/',expand=True)

# df1_1 =df1.dropna(subset=['Month','Year','Day'])

# df1[['Year','Month','Day']] = df1['DATE'].str.split('-',expand=True)

# df1_2 =df1.dropna(subset=['Month','Year','Day'])

# df1=pd.concat([df1_1,df1_2],axis=0,ignore_index=True).fillna("")

# df1.drop('DATE', axis=1, inplace=True)
# df1['DATE']=df1['Month'].astype(str)+'/'+df1['Day']+'/'+df1['Year']
# df1.drop(['Month','Day','Year'], axis=1, inplace=True)


test=pd.concat([df1,df2,df3,df4], axis=0,ignore_index=True).fillna('')

test=test.loc[(test['NAME'] == 'SAN FRANCISCO INTERNATIONAL AIRPORT, CA US')]
test.drop_duplicates(subset=['DATE'])


test['DATE'] = pd.to_datetime(test['DATE'])
test = (test.set_index('DATE')
      .reindex(pd.date_range('07-01-1945', '05-13-2022', freq='D'))
      .rename_axis(['DATE'])
      .fillna("")
      .reset_index())
test.to_csv("SF_complete_daily_summaries.csv")


# norms=pd.read_csv('RedBluff_norms.csv')
# #norms=norms.loc[(norms['STATION'] == 'USW00024257')]
# norms=norms.loc[(norms['NAME'] == 'SAN FRANCISCO INTERNATIONAL AIRPORT, CA US')]

