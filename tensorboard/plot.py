import matplotlib.pyplot as plt
import csv

x = []
y = []

with open("/Users/daniel/MultiHopKG/tensorboard/avgLoss.csv",'r') as f:
    file = csv.reader(f, delimiter=',')
    for row in file:
        x.append(int(row[0]))
        y.append(float(row[1]))

plt.plot(x,y, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss function plot')
plt.legend()
plt.show()