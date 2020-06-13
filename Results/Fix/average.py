import pandas as pd

df1 = pd.read_csv('ACC_CNN5.txt', index_col=0)
print("ACC CNN 5 = ", df1.mean())

df2 = pd.read_csv('LOSS_CNN5.txt', index_col=0)
print("LOSS CNN 5 = ", df2.mean())

df3 = pd.read_csv('ACC_CNN10.txt', index_col=0)
print("ACC CNN 10 = ",df3.mean())

df4 = pd.read_csv('LOSS_CNN10.txt', index_col=0)
print("LOSS CNN 10 = ",df4.mean())


df1 = pd.read_csv('ACC_LSTM5.txt', index_col=0)
print("ACC LSTM 5 = ", df1.mean())

df2 = pd.read_csv('LOSS_LSTM5.txt', index_col=0)
print("LOSS LSTM 5 = ", df2.mean())

df3 = pd.read_csv('ACC_LSTM10.txt', index_col=0)
print("ACC LSTM 10 = ",df3.mean())

df4 = pd.read_csv('LOSS_LSTM10.txt', index_col=0)
print("LOSS LSTM 10 = ",df4.mean())


df1 = pd.read_csv('ACC_CNNLSTM5.txt', index_col=0)
print("ACC CNNLSTM 5 = ", df1.mean())

df2 = pd.read_csv('LOSS_CNNLSTM5.txt', index_col=0)
print("LOSS CNNLSTM 5 = ", df2.mean())

df3 = pd.read_csv('ACC_CNNLSTM10.txt', index_col=0)
print("ACC CNNLSTM 10 = ",df3.mean())

df4 = pd.read_csv('LOSS_CNNLSTM10.txt', index_col=0)
print("LOSS CNNLSTM 10 = ",df4.mean())