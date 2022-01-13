import pandas as pd
import geopandas as geopandas
# import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# In[2]:


pd.set_option('display.max_columns', None)
p = r"G:\python-projects\condas\\testconda\Comparison_table1.csv"
df = pd.read_csv(p, sep=',')

# In[3]:


df['N1'] = df['Normalized_RSSI_1'].apply(lambda a: np.fromstring(re.sub('[{}\[\]]', '', a).split(':')[1], sep=','))
df['N2'] = df['Normalized_RSSI_2'].apply(lambda a: np.fromstring(re.sub('[{}\[\]]', '', a).split(':')[1], sep=','))
df['N3'] = df['Normalized_RSSI_3'].apply(lambda a: np.fromstring(re.sub('[{}\[\]]', '', a).split(':')[1], sep=','))

df['dotN1'] = df['N1'].apply(lambda x: np.dot(x, x))
df['dotN2'] = df['N2'].apply(lambda x: np.dot(x, x))
df['dotN3'] = df['N3'].apply(lambda x: np.dot(x, x))

df['log-dotN1'] = df['dotN1'].apply(lambda x: np.log(x))
df['log-dotN2'] = df['dotN2'].apply(lambda x: np.log(x))
df['log-dotN3'] = df['dotN3'].apply(lambda x: np.log(x))

# In[4]:


a = ['Pred_coord', 'Actual_coord', 'Shift_Point1', 'Shift_Point2', 'Target_RSSI', 'Normalized_RSSI_1',
     'Normalized_RSSI_2', 'Normalized_RSSI_3', 'N1', 'N2', 'N3', '1.9969753918']
df.drop(columns=a, inplace=True)

# In[5]:


focus = df

# In[6]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("dotN1")
sns.scatterplot(y="dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("dotN2")
sns.scatterplot(y="dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("dotN3")

# In[7]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="log-dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("log-dotN1")
sns.scatterplot(y="log-dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("log-dotN2")
sns.scatterplot(y="log-dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("log-dotN3")

# In[8]:


focus.corr()

# In[9]:


f = ((df['dotN3'] < df['dotN1']) & (df['dotN3'] < df['dotN2']))
focus = df[f]

# In[10]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("dotN1")
sns.scatterplot(y="dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("dotN2")
sns.scatterplot(y="dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("dotN3")

# In[11]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="log-dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("log-dotN1")
sns.scatterplot(y="log-dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("log-dotN2")
sns.scatterplot(y="log-dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("log-dotN3")

# In[12]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
sns.histplot(ax=ax, data=focus, x='Accuracy', bins=20, kde=True, hue='Shift_Toward_Point')

# In[13]:


focus['Accuracy'].describe()

# In[14]:


df.corr()

# In[15]:


f = ((df['dotN3'] < df['dotN1']) & (df['dotN3'] > df['dotN2']))
focus = df[f]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("dotN1")
sns.scatterplot(y="dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("dotN2")
sns.scatterplot(y="dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("dotN3")

# In[16]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="log-dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("log-dotN1")
sns.scatterplot(y="log-dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("log-dotN2")
sns.scatterplot(y="log-dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("log-dotN3")

# In[17]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
sns.histplot(ax=ax, data=focus, x='Accuracy', bins=20, kde=True, hue='Shift_Toward_Point')

# In[18]:


focus['Accuracy'].describe()

# In[19]:


focus.corr()

# In[20]:


f = ((df['dotN3'] > df['dotN1']) & (df['dotN3'] < df['dotN2']))
focus = df[f]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("dotN1")
sns.scatterplot(y="dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("dotN2")
sns.scatterplot(y="dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("dotN3")

# In[21]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="log-dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("log-dotN1")
sns.scatterplot(y="log-dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("log-dotN2")
sns.scatterplot(y="log-dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("log-dotN3")

# In[22]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
sns.histplot(ax=ax, data=focus, x='Accuracy', bins=20, kde=True, hue='Shift_Toward_Point')

# In[23]:


focus['Accuracy'].describe()

# In[24]:


focus.corr()

# In[25]:


f = ((df['dotN3'] > df['dotN1']) & (df['dotN3'] > df['dotN2']))
focus = df[f]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("dotN1")
sns.scatterplot(y="dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("dotN2")
sns.scatterplot(y="dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("dotN3")

# In[26]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="log-dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("log-dotN1")
sns.scatterplot(y="log-dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("log-dotN2")
sns.scatterplot(y="log-dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("log-dotN3")

# In[27]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
sns.histplot(ax=ax, data=focus, x='Accuracy', hue='Shift_Toward_Point', bins=20, kde=True)

# In[28]:


focus['Accuracy'].describe()

# In[29]:


focus.corr()

# In[37]:


f = (df['Shift_Toward_Point'] != 'A')
focus = df[f]

# In[38]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.scatterplot(y="dotN1", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[0])
axes[0].set_title("dotN1")
sns.scatterplot(y="dotN2", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[1])
axes[1].set_title("dotN2")
sns.scatterplot(y="dotN3", x="Accuracy", hue='Shift_Toward_Point', data=focus, ax=axes[2])
axes[2].set_title("dotN3")

# In[39]:


focus['Accuracy'].describe()

# In[40]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
sns.histplot(ax=ax, data=focus, x='Accuracy', hue='Shift_Toward_Point', bins=20, kde=True)

# In[41]:


focus.corr()
