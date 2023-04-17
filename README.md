# Ex-06-Feature-Transformation

## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

## PROGRAM:

# DEVELOPED BY: Karpagakirthika.V
# REG NO: 212221220025

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
## OUTPUT:

![image](https://user-images.githubusercontent.com/103020162/232390841-04e744fc-1a51-49f0-847e-3af110350251.png)

![image](https://user-images.githubusercontent.com/103020162/232391010-8b26c530-e51b-4889-b8fe-3403b4242c56.png)

![image](https://user-images.githubusercontent.com/103020162/232392811-a16a2bcf-6f65-41fc-bc89-d5cb7a72c2ee.png)

![image](https://user-images.githubusercontent.com/103020162/232392540-d5683131-2b8f-4ae1-b67b-4e6872100249.png)

![image](https://user-images.githubusercontent.com/103020162/232391974-80985ed6-1154-4d41-8fc3-80f8d2821cc6.png)

![image](https://user-images.githubusercontent.com/103020162/232393518-5e76e24b-2265-494e-8c3e-fdc8136cd084.png)

![image](https://user-images.githubusercontent.com/103020162/232393599-4c193094-5515-4d5b-be91-ed706f3fe5ba.png)

![image](https://user-images.githubusercontent.com/103020162/232393675-ed894212-b748-4ce9-ac01-91c6b87e8e7e.png)

![image](https://user-images.githubusercontent.com/103020162/232394099-19b126a7-6961-4cb6-b156-3a5a72518100.png)

![image](https://user-images.githubusercontent.com/103020162/232394163-62ed1128-104a-4b67-a371-9d503b982419.png)

## RESULT:

Thus feature transformation is done for the given dataset.
