import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

df = pd.read_csv("data/IPL.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(5))




