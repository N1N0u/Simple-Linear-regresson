#linear regression
#transforming my matlab code to python using numpy library
import numpy as np

#feautures are trhe surface of the house
features=np.array([50,80,100,120,150,180,200])
#result is the known price for each house
result=np.array([150,200,250,270,300,350,400])

#Formula y=ax+b
#where a: the slope / b: Intercept / y: result / x: feauters

#the formula to calculate the linear regression is:
# a = (sum (xi-x̄)(yi-ȳ)) / sum (xi-x̄)^2
# b = ȳ - a·x̄

#x̄=mean of feauters
#ȳ=mean of result
n=len(features) #length of the features and result array must be the same
x_bar=np.mean(features) # x̄
y_bar=np.mean(result) # ȳ

X=features-x_bar
Y=result-y_bar  

a= (np.sum(X*Y))/(np.sum(X**2))
print(a)

b=y_bar-(a*x_bar)
print(b)


def predict(feauture):
    return (a*feauture)+b 


predicted_price=predict(130)
print(f"predicted price for a house with surface 130 is: {predicted_price}")





