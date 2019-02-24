
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd 
from datetime import datetime
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import scipy.optimize as sco

# Dow Jones Industrial Average Tickers

# DJIA = ['JNJ','UTX','IBM','CVX','PG','KO','MCD','MSFT','DIS','VZ','NKE','AAPL','INTC', '^GSPC']
SETH = ['BIV','BLV','BND','VCIT','VFIAX','VYM','VO','VB','VWO','VSS','VGTSX','VNQ','PARWX']

DJIA = ['JNJ','IBM','CVX','PG','KO','MCD','MSFT','DIS','BA','NKE','AAPL','INTC','FIZZ','^GSPC']

# Dates
start = datetime(2010, 1, 1)
end = datetime.today()
# Grab data, change to weekly returns and write to CSV


# In[5]:


def getdata(listofstocks, startdate, enddate):
    for i in range(2):
        try:
            x = web.DataReader(listofstocks,"yahoo", startdate, enddate)['Adj Close']
            break
        except Exception as error:
            print("Could not pull the data see below error /n{}".format(error))
    df = pd.DataFrame(x)
    df = df.resample('W-FRI').last().sort_index(ascending=False) #changing data to weekly
    for row in range(len(df)-1):
        df.iloc[row] = df.iloc[row].div(df.iloc[row+1]) #return
    df = df.iloc[:-1]
    df = np.log(df)  #taking log return
    df.to_csv('SethFundData.csv', encoding='utf-8') #write to CSV
    return(df)

data = getdata(DJIA, start, end)
print((data).head(5))
print((data).tail(5))
# print(data[['AAPL']])


# In[6]:


mult = 52.1429 #number of weeks per year
numberofstocks = 13
# data = pd.read_csv('SethFundData.csv', index_col=False)

def split_data(dataframe, numberofstocks):
# 	numberofstocks = numberofstocks + 1
	temp1 = data.iloc[:,0:(numberofstocks)]
	temp2 = data[['^GSPC']]
	temp2.columns = ['S&P']
	df = pd.concat([temp1,temp2],axis = 1) # select number of columns
	sample_df = df.iloc[0:int(len(df.index)/5),:-1]
	trn_df = df.iloc[int(len(df.index)/5):len(df.index),:-1]
	trn_df_graph = df.iloc[int(len(df.index)/5):len(df.index),:]  
	sample_bench = df.iloc[0:int(len(df.index)/5),-1]   
	return(trn_df,sample_df,trn_df_graph, sample_bench)


trn_df = split_data(data,numberofstocks)[0]
trn_df_graph = split_data(data,numberofstocks)[2] 
print((trn_df_graph).head(5))

# df = pd.concat([dataframe.iloc[:,0:(numberofstocks)],datafreme[['^GSPC']]], axis = 1)


# In[7]:


trn_df.mean() * mult


# In[8]:


trn_df.std() * mult


# In[9]:


trn_df.cov() * mult


# In[10]:



def graph_ret(dataframe, location):
	trn_ret_df = dataframe.sort_index(ascending=True)
	trn_ret_df.iloc[:,:] = trn_ret_df.iloc[:,:].add(1)
	for row in range(0,len(trn_ret_df.index), +1):
		trn_ret_df.iloc[row,:] = (trn_ret_df.iloc[row,:]).mul(trn_ret_df.iloc[row-1,:])
# 	trn_ret_df.set_index('Date',inplace=True)
	trn_ret_df.plot(figsize=(14,6))
	plt.savefig(location)
	plt.show()
	plt.close()
	return(trn_ret_df)
print(graph_ret(trn_df_graph, "./return.png"))



# In[11]:


noc=len(trn_df.columns) # total number for index of columns
print("Number of Columns={}".format(noc))


# In[12]:


def basic_ret(dataframe, noc):# annualized mean return
	np.random.seed(0)
	weights = np.random.random(noc) # initial random weights
	weights /= np.sum(weights) #take weight devide by sum of eights and save 
	
	exp_ret = np.dot(dataframe.mean(), weights) * mult #annualized expected return with initial weights
	exp_var = np.dot(weights.T, np.dot(dataframe.cov() * mult, weights))
	exp_vol = np.sqrt(exp_var)
	return(exp_var, exp_vol, exp_ret)


# In[13]:


def simulation(dataframe, steps, noc):
	prets = []
	pvols = []
	for i in range(steps):
		np.random.seed(i)
		weights = np.random.random(noc)
		weights /= np.sum(weights)
		exp_ret = np.dot(dataframe.mean(), weights) * mult
		prets.append(exp_ret)
		exp_var = np.dot(weights.T, np.dot(dataframe.cov() * mult, weights))
		pvols.append(np.sqrt(exp_var))
	prets = np.array(prets)
	pvols = np.array(pvols)
	return(prets, pvols)


# In[14]:


prets, pvols = simulation(trn_df,10000,noc)[0:2]
print("Expected Max Sharp = {}".format(max(prets/pvols)))
exp_vol, exp_ret = basic_ret(trn_df, noc)[1:3]
plt.figure(figsize=(8,4))
plt.scatter(pvols,prets, c= prets / pvols, marker ="o")
plt.plot(exp_vol,exp_ret, 'r*', markersize=15.0)
plt.grid(True)
plt.xlabel("Expected Volatility: Risk")
plt.ylabel("Expected Return")
plt.colorbar(label = "Sharpe ratio")
plt.savefig("./scatterplot.png")
plt.show()
plt.close()


# In[15]:


def statistics(weights):
	weights = np.array(weights)
	pret = np.dot(trn_df.mean(), weights) * mult
	pvol = np.sqrt(np.dot(weights.T, np.dot(trn_df.cov() * mult, weights)))
	return(np.array([pret,pvol, pret / pvol]))


# In[16]:


base_weights = noc * [1/noc,]
print("baseweights = {}".format(np.array(base_weights).round(4)))


# In[17]:


cons = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1}) #defining constraints 
bnds = tuple((0, 1) for x in range(noc))
# stocknames = np.delete(np.array(trn_df.columns),(0), axis=0)
stocknames = np.array(trn_df.columns)
print(cons)
print(bnds)
print(stocknames)


# In[18]:


def min_sharpe(weights):
	return(-statistics(weights)[2])


# In[19]:


opts = sco.minimize(min_sharpe, base_weights, method='SLSQP', bounds=bnds, constraints=cons)
max_sharp_weights = np.column_stack((stocknames, opts['x'].T.round(3)))
max_sharp_weights = np.flipud(max_sharp_weights[max_sharp_weights[:,1].argsort()])
print("Max Sharp weights")
print(max_sharp_weights)
statoutnames = np.array(['return','volatility', 'sharp'])
pd.DataFrame(np.column_stack((statoutnames, statistics(opts['x']).T.round(4))).T)


# In[20]:


# prets, pvols = simulation(trn_df,2500,noc)[0:2]
exp_ret_sharp, exp_vol_sharp = statistics(opts['x'])[0:2]
print(exp_ret_sharp, exp_vol_sharp)
plt.figure(figsize=(12,6))
plt.scatter(pvols,prets, c = prets / pvols, marker ="o")
plt.plot(exp_vol_sharp,exp_ret_sharp, 'r*', markersize=15.0)
plt.grid(True)
plt.xlabel("Expected Volatility: Risk")
plt.ylabel("Expected Return")
plt.colorbar(label = "Sharpe ratio")
plt.savefig("./scatterplot.png")
plt.show()
plt.close()


# In[21]:


def max_ret(weights):
	return(-statistics(weights)[0])


# In[22]:


optr = sco.minimize(max_ret, base_weights, method='SLSQP', bounds=bnds, constraints=cons)
max_ret_weights = np.column_stack((stocknames, optr['x'].T.round(3)))
max_ret_weights = np.flipud(max_ret_weights[max_ret_weights[:,1].argsort()])
print("Maximize return weights")
print(max_ret_weights)
pd.DataFrame(np.column_stack((statoutnames, statistics(optr['x']).T.round(4))).T)


# In[23]:


# prets, pvols = simulation(trn_df,2500,noc)[0:2]
exp_ret_ret, exp_vol_ret = statistics(optr['x'])[0:2]
print(exp_ret_ret, exp_vol_ret)
plt.figure(figsize=(12,6))
plt.scatter(pvols,prets, c = prets / pvols, marker ="o")
plt.plot(exp_vol_ret,exp_ret_ret, 'r*', markersize=15.0)
plt.grid(True)
plt.xlabel("Expected Volatility: Risk")
plt.ylabel("Expected Return")
plt.colorbar(label = "Sharpe ratio")
plt.savefig("./scatterplot.png")
plt.show()
plt.close()


# In[24]:


def min_pvol(weights):   #New function to minimize
	return(statistics(weights)[1])


# In[25]:


optv = sco.minimize(min_pvol, base_weights, method='SLSQP', bounds=bnds, constraints=cons)
min_vol_weights = np.column_stack((stocknames, optv['x'].T.round(3)))
min_vol_weights = np.flipud(min_vol_weights[min_vol_weights[:,1].argsort()])
print("Minimize risk weights")
print(min_vol_weights)
pd.DataFrame(np.column_stack((statoutnames, statistics(optv['x']).T.round(4))).T)


# In[26]:


# prets, pvols = simulation(trn_df,2500,noc)[0:2]
exp_ret_vol, exp_vol_vol = statistics(optv['x'])[0:2]
print(exp_ret_vol, exp_vol_vol)
plt.figure(figsize=(12,6))
plt.scatter(pvols,prets, c = prets / pvols, marker ="o")
plt.plot(exp_vol_vol,exp_ret_vol, 'r*', markersize=15.0)
plt.grid(True)
plt.xlabel("Expected Volatility: Risk")
plt.ylabel("Expected Return")
plt.colorbar(label = "Sharpe ratio")
plt.savefig("./scatterplot.png")
plt.show()
plt.close()


# In[28]:


trets = np.linspace(0.10,0.18,14)
tvols = []
tweights = []

for tret in trets:
	cons = ({'type':'eq', 'fun':lambda x: statistics(x)[0] - tret},   #New constraints
			{'type':'eq', 'fun':lambda x: np.sum(x) - 1})
	res = sco.minimize(min_pvol, base_weights, method='SLSQP', bounds=bnds, constraints=cons)
	tvols.append(res['fun'])
	# temp2 = np.column_stack((res['fun'], res['x'].round(3)))
	# temp1 = np.column_stack(tret)
	
	# tweights.append(temp2)
	# # tweights.append(tret)
	# tweights.append(res['fun'])
	tweights.append(res['x'].round(3))
tvols = np.array(tvols)
tweights = np.array(tweights)


front_line = np.column_stack((trets, tvols)).round(3)
tweights = np.column_stack((front_line,tweights))
tweights = np.transpose(tweights)

dummy = ['return','volatility']
dummy = np.array(dummy)
dummy = np.append(dummy, stocknames)
dummy = np.column_stack((dummy,tweights))
pd.DataFrame(dummy)


# In[30]:


plt.figure(figsize=(8,4))
plt.scatter(pvols,prets, c=prets / pvols, marker ="o")
plt.scatter(tvols,trets, c=trets / tvols, marker ="x")
plt.plot(statistics(opts['x'])[1],statistics(opts['x'])[0], 'r*', markersize=15.0)
plt.plot(statistics(optv['x'])[1],statistics(optv['x'])[0], 'y*', markersize=15.0)
# plt.plot(statistics(optr['x'])[1],statistics(optr['x'])[0], 'g*', markersize=15.0)
plt.grid(True)
plt.xlabel("Expected Volatility: Risk")
plt.ylabel("Expected Return")
plt.colorbar(label = "Sharpe ratio")
plt.savefig("./FrontierLine.png")
plt.show()
plt.close()


# In[32]:


sample_df = split_data(data, numberofstocks)[1]
sample_df.head(10)


# In[33]:


sample_df.mean() * mult


# In[34]:


sample_df.std() * mult


# In[35]:


##Choose your column
xx = 11
optweights = tweights[:,xx - 1][2:]


# In[36]:


def test(weights, dataframe):
	weights = np.array(weights)
	pret = np.dot(dataframe.mean(), weights) * mult
	pvol = np.sqrt(np.dot(weights.T, np.dot(dataframe.cov() * mult, weights)))
	return(np.array([pret,pvol, pret / pvol]))


# In[37]:


pd.DataFrame(np.column_stack((statoutnames, test(optweights,sample_df).T.round(4))).T)


# In[38]:


sample_banch = trn_df_graph = split_data(data,numberofstocks)[3]
sample_banch.mean() * mult
teststats = np.array([sample_banch.mean() * mult, sample_banch.std() * mult, ((sample_banch.mean() * mult) / (sample_banch.std() * mult))])
pd.DataFrame(np.column_stack((statoutnames, teststats.T.round(5))).T)


# In[39]:


def max_test(weights,dataframe):
	return(test(weights,dataframe)[1])

test_cons = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1}) #defining constraints 



# In[40]:


opt_test = sco.minimize(max_test, base_weights,args=(sample_df), method='SLSQP', bounds=bnds, constraints=test_cons)
min_vol_test = np.column_stack((stocknames, opt_test['x'].T.round(3)))
min_vol_test = np.flipud(min_vol_test[min_vol_test[:,1].argsort()])
print("Minimize risk weights")
print(min_vol_test)
# pd.DataFrame(np.column_stack((statoutnames, test(opt_test['x'],).T.round(4))).T)

