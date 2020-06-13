import pandas as pd

print("\n")
print("CNNLSTM")
df1 = pd.read_csv('run-cnnlstmmodel-tag-Accuracy.csv', index_col=0)
print(df1.mean())
df1 = pd.read_csv('run-cnnlstmmodel-tag-Accuracy_Validation.csv', index_col=0)
print(df1.mean())

print("\n")
print("CNN")
df1 = pd.read_csv('run-cnnmodel-tag-Accuracy.csv', index_col=0)
print(df1.mean())
df1 = pd.read_csv('run-cnnmodel-tag-Accuracy_Validation.csv', index_col=0)
print(df1.mean())

print("\n")
print("LSTM")
df1 = pd.read_csv('run-lstmmodel-tag-Accuracy.csv', index_col=0)
print(df1.mean())
df1 = pd.read_csv('run-lstmmodel-tag-Accuracy_Validation.csv', index_col=0)
print(df1.mean())