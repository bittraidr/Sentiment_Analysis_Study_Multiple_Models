# Sentiment_Analysis_Study_Multiple_Models
UC Berkeley Fintech: Group project using multiple ML models to determine market sentiment

### Required Installs
pip install textblob<br>
pip install vaderSentiment

### Model Prediction Results
**Baseline investment strategy with SVC classifier:**<br>
SVC is well suited for binary classification of sentiment analysis<br>
<img src="Resources/Baseline.png" alt="Baseline" width="600"/>

**Investment strategy with SVC classifier and sentiment analysis:**<br>
SVC is well suited for binary classification of sentiment analysis<br>
<img src="Resources/SVC.png" alt="SVC with sentiment data" width="600"/>

**Investment strategy with LDA classifier and sentiment analysis:**<br>
LDA is well suited for linear and binary data; however, it adds a dimentionality reduction feature to classification of sentiment analysis, which negatively impacted the accuracy of the predictions<br>
<img src="Resources/LDA.png" alt="LDA with sentiment data" width="600"/>

### Presentation
[Project 2](Project_2-Group_2.pptx)

### Files
[Main Code](main.ipynb)<br>
[Models References](Code/SVC_LDA_models.py)<br>
[Data Construction](Code/data_pre-processing.ipynb)<br>
[SVC Model](Code/SVC_with_sentiment.ipynb)<br>
[LDA Model](Code/LDA_with_sentiment.ipynb)<br>

### References
[Sentiment Analysis](https://www.youtube.com/watch?v=4OlvGGAsj8I)
