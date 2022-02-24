import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data1=pd.read_csv('/Users/harshittiwari/Downloads/HEROMOTOCO.NS.csv')

data2=pd.read_csv('/Users/harshittiwari/Downloads/TTM.csv')

data3=pd.read_csv('/Users/harshittiwari/Downloads/RAYMOND.NS.csv')

df1=data1[['Date','Adj Close']]
df2=data2[['Date','Adj Close']]
df3=data3[['Date','Adj Close']]

df1=df1.set_index('Date')
df2=df2.set_index('Date')
df3=df3.set_index('Date')
#print(df3.head())

stocks=pd.concat([df1,df2,df3],axis=1)
stocks.columns=['HEROMOTOCO','TTM','RAYMOND']

log_ret=np.log(stocks/stocks.shift(1))
print(log_ret.head(n=400))

np.random.seed(42)
NumPorts=2000
AllWeights=np.zeros((NumPorts,len(stocks.columns)))
RetArr=np.zeros(NumPorts)
VolArr=np.zeros(NumPorts)
SharpeArr=np.zeros(NumPorts)

for x in range(NumPorts):
    weights=np.array(np.random.random(3))
    weights=weights/np.sum(weights)
    AllWeights[x,:]=weights
    RetArr[x]=np.sum((log_ret.mean()*weights*249))
    VolArr[x]=np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*249,weights)))
    SharpeArr[x]=RetArr[x]/VolArr[x]
    

print('Max sharpe ratio:{}'.format(SharpeArr.argmax()))


print(AllWeights[460:,])
MaxSrRet=RetArr[SharpeArr.argmax()]
MaxSrVol=VolArr[SharpeArr.argmax()]


plt.figure(figsize=(12,8))
plt.scatter(VolArr,RetArr,c=SharpeArr,cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(MaxSrVol,MaxSrRet,c='red',s=50) # red dot
plt.show()
