import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv('C:/Users/TheRotaryFox/Documents/ComputerVision/IsolationForestTest/Sample scores.csv')
pf = pd.read_csv('C:/Users/TheRotaryFox/Documents/ComputerVision/sampleSet.csv')
print(df)
model=IsolationForest(n_estimators=10, max_samples='auto', contamination=float(0.2), max_features=1.0)
model.fit(df[['Scores']])
pf['scores']=model.decision_function(pf[['Scores']])
pf['anomaly']=model.predict(pf[['Scores']])
print(pf.head(100))


ypoints = pf[['Scores']]
plt.plot(ypoints)
plt.show()

x = df[['Scores']]
plt.hist(x)
plt.show() 

outliers_counter = len(pf[pf['anomaly']==-1])
print(outliers_counter)