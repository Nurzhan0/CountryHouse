import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats

#csv = pd.read_csv('coun_sep-feb.csv',sep=',', parse_dates=True, header=0, encoding='utf-8')  # OK
csv = pd.read_csv('C:/Users/user/Desktop/диссертация/test/coun_sep-feb.csv',sep=',', header=0, encoding='utf-8')  # OK

dfp1 = csv.iloc[:260497].copy() # Until here data is in format m-d-Y
#line 291785 is another header, so it is discarded
dfp2 = csv.iloc[260498:].copy() # From here onwards date is in format d-m-Y


# In[] Correct date and time, join dataframes

print('')
print('Data loaded 1st part - tail:',dfp1.tail())
print('Data loaded 2nd part - head:',dfp2.head())

# Join date and time in the same string
dfp1['Date Time'] = dfp1['Date']+' '+dfp1['Time']
dfp2['Date Time'] = dfp2['Date']+' '+dfp2['Time'] #+'+0500'

# Convert date and time from sring to datetime
dfp1['Datetime'] = pd.to_datetime( dfp1['Date Time'], format='%m/%d/%Y %H:%M:%S' )#.map(lambda x: x.tz_convert('Asia/Almaty') #.astimezone('Asia/Almaty') # - pd.Timedelta('05:00:00')
dfp2['Datetime'] = pd.to_datetime( dfp2['Date Time'], format='%d.%m.%Y %H:%M:%S' )#.map(lambda x: x.tz_convert('Asia/Almaty') #- pd.Timedelta('05:00:00')

dfp1 = dfp1.drop('Date Time',axis=1)
dfp2 = dfp2.drop('Date Time',axis=1)

#df1.set_index('Datetime')
#df2.set_index('Datetime')

# In[]

# Merge into one dataframe
df = pd.concat([dfp1,dfp2])
df = df.sort_values(by='Datetime')
df.set_index('Datetime')


# In[]
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Data Cleaning
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''' plot clean and original data '''
def original_clean_plot(df,dfc,var):
    plt.plot(df['Datetime'], df[var],label = 'Original')
    plt.plot(dfc['Datetime'], dfc[var],label = 'Cleaned')
    plt.legend()
    plt.title(var)
    plt.show()

''' show statistics for one variable '''
def show_stats(df, var):
    print('\nStats for ',var)
    print('Minimum Maximum Mean STD')
    print(np.min(df[var].astype(float)), np.max(df[var].astype(float)) )

''' Clean all variables in the dataset and store result in new dataframe'''
def clean_data(df):
    print('\n\nCleaning data:')
    dfnumeric = df.copy()  # Dataframe cleaned from non-numeric 
    dfc = df.copy()  # Dataframe cleaned from non-numeric and discrepants
    for c in df.columns:
        if c=='Date':
            continue
        if c=='Time':
            continue
        if c=='Datetime':
            continue
        print('Cleaning ',c)
        
        mask = pd.to_numeric(df[c], errors='coerce').isna()
        # Replace non-numerics by nan
        dfc[c].iloc[mask] = np.nan
        dfnumeric[c].iloc[mask] = np.nan
        dfnumeric[c] = dfnumeric[c].astype(float)
        dfc[c] = dfc[c].astype(float)
        z = stats.zscore(dfc[c],nan_policy='omit')
        # replace all samples where z > 3 by nan
        dfc[c][ np.abs(z) > 1.5 ] = np.nan
        dfc[c]=dfc[c].interpolate(method='linear')
    return dfnumeric, dfc

dfnumeric, dfcleaned = clean_data(df) # Dataframes cleaned from non-numeric and from discrepants too

# In[] Show plots

original_clean_plot(dfnumeric,dfcleaned,'Current')
plt.figure()
original_clean_plot(dfnumeric,dfcleaned,'Humidity outside')
plt.figure()
original_clean_plot(dfnumeric,dfcleaned,'Dew-point outside')


# In[] Show statistical values

show_stats(dfnumeric,'Temperature')
show_stats(dfcleaned,'Temperature')
show_stats(dfcleaned,'Dew-point outside')

# In[] Show correlation matrix

corr = dfcleaned.corr()

# In[] PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dfpca =dfcleaned.drop( ['Date', 'Time', 'Datetime', 'Temperature', 'Z'],axis=1 )
dfpca = dfpca.dropna()

scaler = StandardScaler()
dfpcasc = scaler.fit_transform(dfpca)

pca = PCA(n_components=4)
comp = pca.fit_transform(dfpcasc)

plt.scatter(comp[:,0],comp[:,1])
plt.title('Data distribution')
plt.xlabel('First Component')
plt.ylabel('Second Component')


# In[] k-means

from sklearn.cluster import KMeans

model = KMeans(n_clusters=2)
model.fit(comp)

print('Centróides:\n',model.cluster_centers_)

plt.figure()
plt.scatter(comp[:,0], comp[:,1], c=model.labels_)




# In[] DBSCAN

from sklearn.cluster import DBSCAN
import pickle

# x: Data, y: Labels

model = DBSCAN(eps=1.1, min_samples=5)

# I'll get just a slice of the dataset to make dbscan faster
l = int(len(comp) / 3 * .9)
chunk = comp[l : -l]




model.fit(chunk)
## In[] Save into pickle
import pickle
fname = 'C:/Users/user/Desktop/диссертация/test/country-house-dbscan-2.bin'
file = open(fname, 'wb')
# dump information to that file
pickle.dump(model, file)
# close the file
file.close()

# In[] Cluster analysis
''' 
import pickle
file = open(fname, 'rb')
# dump information to that file
model = pickle.load(file)
# close the file
file.close()
''' 


print('Predições:\n',model.labels_)

plt.figure()
plt.scatter(chunk[:,0], chunk[:,1], c=model.labels_)


## In[] Cluster analysis

nclusters = set(model.labels_)

print('Number of clusters formed: ',len(nclusters)-1)
for j in nclusters:
    print('  Points in cluster ',j,
          ': ',len(model.labels_[model.labels_==j ]))

print('Out of clusters: ', len(model.labels_[  model.labels_==-1  ]) )


# In[] statistical parameters per cluster

chunk = dfpca[l : -l]

for j in nclusters:
    print('  Cluster ',j)
    cluster = chunk [ model.labels_==j  ]
    print(cluster.describe())

cluster1 = chunk [ model.labels_==0  ]
cluster1data = cluster1.describe()

cluster2 = chunk [ model.labels_==1  ]
cluster2data = cluster2.describe()

cluster3 = chunk [ model.labels_==2  ]
cluster3data = cluster3.describe()

nocluster = chunk [ model.labels_==-1  ]
noclusterdata = nocluster.describe()

