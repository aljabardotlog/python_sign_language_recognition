import matplotlib.pyplot as plt
import numpy as np

x,y= np.loadtxt('ACC_CNNLSTM5.txt', delimiter=',', unpack=True)


plt.figure(figsize=(17,10))
plt.plot(x,y, marker='o')

plt.title('ACC_CNNLSTM5')

plt.xlabel('Epoch Result')
plt.ylabel('Value')

plt.show()
