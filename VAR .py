#!/usr/bin/env python
# coding: utf-8

# In[7]:


import VAR
from VAR import *


# In[10]:


rawdata= pd.read_excel('/Users/derek/Desktop/education/School/graduate/lesson/6282/HW2/MacroData.xlsx',sheet_name='Data')
rawdata=rawdata.set_index('Unnamed: 0')
rawdata['PE']=rawdata.loc[:,'P']/rawdata.loc[:,'E']
rawdata['DY']=rawdata.D/12/rawdata.P
rawdata['logPE']=np.log(rawdata.PE)
rawdata['logCPI']=np.log(rawdata.CPI)
rawdata['logDY']=np.log(1+rawdata.DY)
rawdata['DlogPE']=rawdata.logPE.diff()
rawdata['DlogCPI']=rawdata.logCPI.diff()
rawdata['logDY']=np.log(1+rawdata.DY)
rawdata['logRF']=np.log(1+rawdata.RF/100)
rawdata['logMktRet']=np.log((rawdata.P+rawdata.D)/rawdata.P.shift())
rawdata['logRealEG']=rawdata.logMktRet-rawdata.DlogPE-rawdata.DlogCPI--rawdata.logDY
data2run=rawdata.loc[:,['DlogPE','DlogCPI','logRealEG','logDY','logRF','HML','SMB']]
data2run=data2run.dropna()
data2run=data2run.dropna()
lags= 2
estimation=VARest(data2run,lags)
estimation


# In[25]:


pi= tt(estimation['PiPrime'])
omega=(estimation['Omega'])
def make_err(omega):
    n=omega.shape[0]
    e= np.random.normal(size=n)
    e=np.reshape(e,(-1,1))
    epsilon=np.matmul(np.linalg.cholesky(omega),e)
    
    #chol=np.linalg.cholesky(omega)
    #epsilon=chole@e
    return (epsilon)




# In[35]:



simdata=data2run.copy()
horizon= 120
for i in range (horizon):
    epsilon= make_err(omega)
    z=np.array([[1]])#[1]inside just the [1] we created
    for ilag in range(lags):
        alag= simdata.iloc[ilag-1,:]
        z=np.concatenate((z,np.reshape(alag.values,(-1,1))),axis=0)
    y=pi@z+ epsilon
    newdata= simdata.index[-1]+pd.Timedelta(1,'M')
    newrow=pd.DataFrame(tt(y),columns=simdata.columns, index=[newdata])
    simdata= simdata.append(newrow)
simdata.head()


# In[32]:





# In[ ]:





# In[ ]:





# In[ ]:




