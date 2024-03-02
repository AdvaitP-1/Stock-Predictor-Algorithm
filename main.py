# Basic Plotting, Math, and file handling imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

# To ignore warnings 
import warnings 
warnings.filterwarnings("ignore")

# Read data
df = pd.read_csv("C:/Projects/Python/Stock Predictor/NVDA.csv")

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract quarter end from the 'Date' column
df['Quarter_end'] = df['Date'] + pd.offsets.QuarterEnd(0)

# Ensure 'Date' column is in string format
df['Date'] = df['Date'].astype(str)

# Extract additional date features
splitted = df['Date'].str.split('-', expand=True)  # assuming the date format is YYYY-MM-DD
df['day'] = splitted[2].astype('int')
df['month'] = splitted[1].astype('int')
df['year'] = splitted[0].astype('int')

# Display basic statistics and information about the data
print(df.describe())
print("-----------------------------------------------")
print(df.info())

# Continue with your analysis, visualization, and any other operations

# Plotting close price
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Nvidia Close Price.', fontsize=15)
plt.ylabel('Price($)')
plt.show()

# Drop redundant column
df = df.drop(['Adj Close'], axis=1)

# Check for null values
print(df.isnull().sum())

# Visualize distribution and boxplot
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.histplot(df[col])
plt.show()

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxplot(df[col])
plt.show()

# Extract additional date features
splitted = df['Date'].str.split('[^\w]+', expand=True)  # Split using any non-word character
df['day'] = splitted[2].astype('int')
df['month'] = splitted[1].astype('int')
df['year'] = splitted[0].astype('int')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract quarter end from the 'Date' column
df['Quarter_end'] = df['Date'] + pd.offsets.QuarterEnd(0)

# Group by quarter end and visualize
print(df.groupby('Quarter_end').mean())
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
plt.show()

# Feature engineering
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Visualize target distribution
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.show()

# Visualize correlation heatmap
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

# Prepare features and target
features = df[['open-close', 'low-high', 'Quarter_end']]
target = df['target']

# Scale features
scaler = StandardScaler()
features[['open-close', 'low-high']] = scaler.fit_transform(features[['open-close', 'low-high']])

# Split data into train and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=42)


# Train models
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
for model in models:
    model.fit(X_train, Y_train)
    print(f'{model}:')
    print('Training Accuracy:', metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:,1]))
    print('Validation Accuracy:', metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:,1]))
    print()

# Plot confusion matrix
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()