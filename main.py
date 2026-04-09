from stock_sim import stock
import matplotlib.pyplot as plt

stock_1 = stock(100,0.05,0.2,"",{"lambda":0.5,"mu":0.1,"sigma":0.1})
prices = stock_1.simulate()


plt.figure(figsize=(10,5))
plt.plot(prices)
plt.title("Simulated Stock Prices")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.grid(True)
plt.show()