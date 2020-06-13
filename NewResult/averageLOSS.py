import pandas as pd

print("\n")
print("CNNLSTM")
df1 = pd.read_csv('run-cnnlstmmodel-tag-Loss.csv', index_col=0)
print(df1.mean())
df1 = pd.read_csv('run-cnnlstmmodel-tag-Loss_Validation.csv', index_col=0)
print(df1.mean())

print("\n")
print("CNN")
df1 = pd.read_csv('run-cnnmodel-tag-Loss.csv', index_col=0)
print(df1.mean())
df1 = pd.read_csv('run-cnnmodel-tag-Loss_Validation.csv', index_col=0)
print(df1.mean())

print("\n")
print("LSTM")
df1 = pd.read_csv('run-lstmmodel-tag-Loss.csv', index_col=0)
print(df1.mean())
df1 = pd.read_csv('run-lstmmodel-tag-Loss_Validation.csv', index_col=0)
print(df1.mean())