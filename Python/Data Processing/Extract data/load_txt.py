import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('web_traffic.txt', delimiter = '\t')
# Check data
#print(data[:10])
# Check data format
#print(data.shape)
# Preprocessing and clean data
X=data[:,0]
Y=data[:,1]
#print(X)
#print(Y)
# Check 'NaN' value
err_dt=np.sum(np.isnan(Y)) 
#print(err_dt)
#Clear err_dt
X = X[~np.isnan(Y)]
Y = Y[~np.isnan(Y)]
plt.scatter(X,Y)
plt.title("Luu luong truy cap web")
plt.xlabel("Thoi gian")
plt.ylabel("Truy cap/gio")
plt.xticks([w*7*24 for w in range(5)],['Tuan %i'%(w+1) for w in range(5)])
plt.autoscale(tight=True)
plt.grid()
plt.show()