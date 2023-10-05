#[Joe] 1. Acquire and process stock data
#[Alejandra]x 2. Acquire and process sentiment analysis data
#[Session] 3. Run baseline
#[Joe]x 4. Run sentiment analysis with LinearDiscreminateAnalysis
#[Edward] 5. Run sentiment analysis with alternate classifier
#[Edward] 6. Combine results
#[Session] 7. Put together PPT preso

import pandas as pd
import numpy as np
from pathlib import Path
from textblob import TextBlob
# import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import yfinance as yf
import warnings
from pandas.tseries.offsets import DateOffset
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Ignore all warnings within this code block
warnings.filterwarnings("ignore")

if __name__=='__main__': 
    data_path='../Resources/'
else: 
    data_path='Resources/'

## Acquire and process sentiment data from Kaggle
sentiment_df=pd.read_csv(Path(f'{data_path}combined_csv.csv'))
# sentiment_df.head()

# obtain polarity and subjectivity scores (potentially factor volume of information in the sentiment analysis)

# create a function to calculate the subjectivity
def calculate_subjectivity(headlines):
    return TextBlob(headlines).sentiment.subjectivity

# create a function to calculate the subjectivity
def calculate_polarity(headlines):
    return TextBlob(headlines).sentiment.polarity

# create two new columns "Subjectivity" and "Polarity"
sentiment_df["Subjectivity"] = sentiment_df["Headline"].apply(calculate_subjectivity)
sentiment_df["Polarity"] = sentiment_df["Headline"].apply(calculate_polarity)

# create function to get the sentiment scores 
def get_scores(headlines):
    get_score= SentimentIntensityAnalyzer()
    sentiment=get_score.polarity_scores(headlines)
    return sentiment

# get daily sentiment scores
compound = []
neg = []
pos = []
neu = []
score = 0

for x in range(0, len(sentiment_df["Headline"])):
    score = get_scores(sentiment_df["Headline"][x])
    compound.append(score["compound"])
    neg.append(score["neg"])
    neu.append(score["neu"])
    pos.append(score["pos"])
    
# Add Column with sentiment scores
sentiment_df["compound"]= compound
sentiment_df["neg"]= neg
sentiment_df["pos"]= pos
sentiment_df["neu"]= neu

# Assuming you have a DataFrame named 'sentiment_df'
sentiment_df.rename(columns={'Volume.1': 'Tvolume'}, inplace=True)

#Display Dataframe
# sentiment_df.head()

# Calculate daily returns 
sentiment_df["daily returns"]= sentiment_df["TSLA Close"].pct_change()

# Drop SP500 Columns 
sentiment_df.drop(columns=["SP500 Close", "Volume"], inplace = True)

# Set Time to be the index and convert to date time format
sentiment_df= sentiment_df.set_index("Time")
sentiment_df.index= pd.to_datetime(sentiment_df.index)

# Calculations for model with SP500 
# Calculate daily returns 
# sentiment_df["daily returns"]= sentiment_df["SP500 Close"].pct_change()
# sentiment_df.drop(columns=["TSLA Close", "Tvolume"], inplace = True)
# sentiment_df= sentiment_df.set_index("Time")
# sentiment_df.index= pd.to_datetime(sentiment_df.index)

# create label column add 1 when daily returns is positive and 0 when it is negative
# sentiment_df["label"]= 0
# sentiment_df.loc[(sentiment_df["daily returns"]> 0), 'label'] = 1

# # verify that label is int
# sentiment_df["label"].dtype

# # calculate SMA short and SMA slow
sentiment_df["sma_short"]= sentiment_df.rolling(7)["TSLA Close"].mean()
sentiment_df["sma_long"]= sentiment_df.rolling(60)["TSLA Close"].mean()
sentiment_df.dropna(inplace=True)

# calculate SMA short and SMA slow
# sentiment_df["sma_short"]= sentiment_df.rolling(7)["SP500 Close"].mean()
# sentiment_df["sma_long"]= sentiment_df.rolling(30)["SP500 Close"].mean()
# sentiment_df.dropna(inplace=True)

# display dataframe
# sentiment_df.columns

# create features variable with columns for X
#TSLA Version
features=['Tvolume', 'Subjectivity', 'Polarity',
       'compound', 'neg', 'pos', 'neu', 'sma_short',
       'sma_long']

X = sentiment_df[features].shift().dropna().copy()
# X.tail()

#SP500 Version
# create features variable with columns for X
# features=['Volume', 'Subjectivity', 'Polarity',
#        'compound', 'neg', 'pos', 'neu', 'sma_short',
#        'sma_long']

# X = sentiment_df[features].shift().dropna().copy()
# X.tail()

# create labels for y value using daily returns
y=(sentiment_df['daily returns']>0).astype(int)[1:]

# verify x and y are same length
# display(len(X))
# display(len(y))

# Select the start of the training period
training_begin = X.index.min()

# Display the training begin date
# print(training_begin)

# Select the ending period for the training data with an offset of 3 months
training_end = X.index.min() + DateOffset(months=3)

# Display the training end date
# print(training_end)

# Generate the X_train and y_train DataFrames
X_train = X.loc[training_begin:training_end]
y_train = y.loc[training_begin:training_end]

# Review the X_train DataFrame
# X_train.head()

# Generate the X_test and y_test DataFrames
X_test = X.loc[training_end+DateOffset(hours=1):]
y_test = y.loc[training_end+DateOffset(hours=1):]

# Review the X_test DataFrame
# X_train.head()

## Scale the features DataFrames

# Create a StandardScaler instance
scaler = StandardScaler()

# Apply the scaler model to fit the X-train data
X_scaler = scaler.fit(X_train)

# Transform the X_train and X_test DataFrames using the X_scaler
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

## Use a classifier to predict future results

# From SVM, instantiate SVC classifier model instance
svm_model = svm.SVC()
 
# Fit the model to the data using the training data
svm_model = svm_model.fit(X_train_scaled, y_train)
 
# Use the testing data to make the model predictions
svm_pred = svm_model.predict(X_test_scaled)

# Review the model's predicted values
# svm_pred[:10]

# Use a classification report to evaluate the model using the predictions and testing data
svc_testing_report = classification_report(y_test, svm_pred)

# Print the classification report
# print(svc_testing_report)

# Create a predictions DataFrame
predictions_sentiment_df = pd.DataFrame(index=X_test.index)

# Add the SVM model predictions to the DataFrame
predictions_sentiment_df['Predicted'] = svm_pred

# Add the actual returns to the DataFrame
predictions_sentiment_df['Actual Returns'] = sentiment_df['daily returns']

# Add the strategy returns to the DataFrame
predictions_sentiment_df['Strategy Returns'] = predictions_sentiment_df['Actual Returns'] * predictions_sentiment_df['Predicted']

# Review the DataFrame
# display(predictions_sentiment_df.head())
# display(predictions_sentiment_df.tail())

# count predictions values and test values
# print("Predictions value counts:")
# print(f"{predictions_sentiment_df['Predicted'].value_counts()}")
# print("Y test value counts:")
# print(f"{y_test.value_counts()}")

# plot cumulative returns
SVC_plot_cum_returns= (1 + predictions_sentiment_df[["Actual Returns", "Strategy Returns"]]).cumprod()
# SVC_plot_cum_returns.plot(title='SVM Model')

# Save the plot as a png file
plt.savefig(Path(f'{data_path}SVC.png'), format='png')

# Save the plot as a JPG file
# plt.savefig('SP500_SVM.jpg', format='jpg')

# instantiate LinearDiscriminantAnalysis
lda_model = LinearDiscriminantAnalysis()

# Fit the model to the data using the training data
linear_disc_model = lda_model.fit(X_train_scaled, y_train)
 
# Use the testing data to make the model predictions
linear_disc_model = lda_model.predict(X_test_scaled)

# Review the model's predicted values
linear_disc_model[:10]

# Use a classification report to evaluate the model using the predictions and testing data
linear_disc_testing_report = classification_report(y_test, linear_disc_model)

predictions_sentiment_3_df = pd.DataFrame(index=X_test.index)

# Add the SVM model predictions to the DataFrame
predictions_sentiment_3_df['Predicted'] = linear_disc_model

# Add the actual returns to the DataFrame
predictions_sentiment_3_df['Actual Returns'] = sentiment_df['daily returns']

# Add the strategy returns to the DataFrame
predictions_sentiment_3_df['Strategy Returns'] = predictions_sentiment_3_df['Actual Returns'] * predictions_sentiment_3_df['Predicted']

# Review the DataFrame
# display(predictions_sentiment_3_df.head())
# display(predictions_sentiment_3_df.tail())

# count predictions values and test values
# print("Predictions value counts:")
# print(f"{predictions_sentiment_df_2['Predicted'].value_counts()}")
# print("Y test value counts:")
# print(f"{y_test.value_counts()}")

# plot cumulative returns
LDA_plot_cum_returns= (1 + predictions_sentiment_3_df[["Actual Returns", "Strategy Returns"]]).cumprod()
# LDA_plot_cum_returns.plot(title='Linear Discriminant Analysis Model')

# Save the plot as a png file
plt.savefig(Path(f'{data_path}LDA.png', format='png'))

# predictions_sentiment_df_2[["Actual Returns","Strategy Returns"]].plot()
# # plt.savefig('SP500 LinearDiscriminantAnalysis.jpg', format='jpg')
# plt.savefig('TSLA LinearDiscriminantAnalysis.jpg', format='jpg')
