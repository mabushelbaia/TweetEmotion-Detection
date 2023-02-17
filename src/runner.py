import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from model import procces_tweet
# Load the saved model
model = joblib.load('../models/RandomFortress.joblib')
tfidf = joblib.load('../models/features.joblib')
option = input("Enter 1 to test a tweet/s or 2 to test a file: ")
if option == '1':
    while True:
        text = input("Enter your tweet: ")
        text = procces_tweet(text)
        print(text)
        X_test = tfidf.transform([text])

        # Make predictions on the test data
        y_pred = model.predict(X_test)
        print(y_pred)
else:
    file = input("Enter the file name: ")
    df = pd.read_csv(file, sep='\t', header=None, names=['Label', 'Tweet'])
    df['Filtered_Tweet'] = df['Tweet'].apply(procces_tweet)
    x = df['Filtered_Tweet'].values
    y = df['Label'].values
    x_feat = tfidf.transform(x)
    y_pred = model.predict(x_feat)
    for i in range(len(y_pred)):
        # print("Tweet: ", df['Tweet'][i])
        print("Predicted: ", y_pred[i])
        print("Actual: ", y[i])
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm)