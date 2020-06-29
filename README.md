# README

Achieve some Logistic-Regression solver with `numpy`. Compare different optimization algorithm in those model, waiting for more optimization algorithm.

Use `sklearn.datesets` to generate training data. 

## Logistic-Regression

Logistic Regression with difference optimization algorithm. 

**RUN EXAMPLE:** 

+ Gauss-Newton iteration (GN): `python logistic.py GN`
+ Gradient Descent (GD): `python logistic.py GD --learning_rate=0.001 --iteration=500`
+ Stochastic Gradient Descent (SGD): `python logistic.py SGD --learning_rate=0.01`
+ Mini Batch Gradient Descent (MBGD): `python logistic.py MBGD --learning_rate=0.001 --iteration=50 --batch_size=20`

