import linear01
import  matplotlib.pyplot as plt
new_temperatures = linear01.pd.DataFrame(linear01.np.array([26, 30]))
predicted_sales = linear01.lm.predict(new_temperatures)
print(predicted_sales)
plt.scatter(linear01.temperatures, linear01.drink_sales)  # Ã¸ÂI
regression_sales = linear01.lm.predict(linear01.X)
plt.plot(linear01.temperatures, regression_sales, color="blue")
plt.plot(new_temperatures, predicted_sales,color="red", marker="o", markersize=10)
plt.show( )
