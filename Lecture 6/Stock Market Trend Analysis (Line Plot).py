import numpy as np
import matplotlib.pyplot as plt

#Simulated stock prices
days= np.arange(1,31)
prices= np.cumsum(np.random.randn(30) * 2 + 100)

plt.figure(figsize=(10,5), dpi=100)
plt.plot(days, prices, linestyle='-', marker = "o" ,
         color='blue', markersize=6),
label= "Stock Price"

#Adding tables, titles and legends
plt.xlabel("Days"),
plt.ylabel("Stock Price(USD)"),
plt.title("Stock Price Trend Analysis"),
plt.legend(),
plt.grid(True),

plt.show()