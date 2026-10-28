# Module 02 · Regression

**ShopWave asks:** *"How much will each customer spend next month? Marketing wants to know who to send the premium catalogue to."*

**Time:** about 14 hours. **Prerequisites:** Module 01 (Pandas, plotting, mean and standard deviation).

## Where you are

Marketing has a budget for 2,000 printed catalogues and 40,000 customers. They want to send them to the people most likely to spend. Nobody can look at 40,000 customers one by one, and a rule like "send it to everyone who bought last month" ignores most of what you know.

What you need is a function: *customer facts in, expected spend out*. Learning that function from past data is called **regression**, and it is the simplest and most useful kind of machine learning. Everything later in the course (classification, neural networks, even language models) is a variation on what you learn here.

## What you will be able to do afterwards

- Explain what a model is: features in, prediction out, parameters learned from data.
- Fit a straight line to data by hand with the least-squares formulas, and reproduce the result in NumPy and scikit-learn.
- Measure how wrong a model is with MAE, MSE, RMSE and R², and know which one to report to whom.
- Explain gradient descent as "walking downhill", and implement it in ten lines.
- Use many features at once, add curves with polynomial features, and spot the first signs of overfitting.
- Predict with a decision tree and explain when trees beat lines.
- Build a working spend-prediction model for ShopWave.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 02.1 | [What is a model? Features, target, prediction, error](01-what-is-a-model/) | 1 h |
| 02.2 | [Linear regression by hand: the best straight line](02-linear-regression-by-hand/) | 2 h |
| 02.3 | [Measuring error: MAE, MSE, RMSE, R²](03-measuring-error/) | 1.5 h |
| 02.4 | [Gradient descent: walking downhill](04-gradient-descent/) | 2 h |
| 02.5 | [Many features: multiple linear regression](05-multiple-regression/) | 1.5 h |
| 02.6 | [Curves: polynomial regression and overfitting](06-polynomial-and-overfitting/) | 1.5 h |
| 02.7 | [Decision tree regressor](07-decision-tree-regressor/) | 1.5 h |
| 02.P | [Mini-project: predict customer spend](project/) | 3 h |

## The maths in this module

Only four ideas, each taught when needed: the **mean**, the **slope** of a line, **squaring** an error so negatives do not cancel, and the **derivative** as "which way is downhill". If you can compute an average, you can do all of it.
