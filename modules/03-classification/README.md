# Module 03 · Classification

**ShopWave asks:** *"Which customers are about to leave? Retention has budget for 500 win-back vouchers and wants to send them to the right people."*

**Time:** about 16 hours. **Prerequisites:** Module 02 (features, target, fitting a model, measuring error).

## Where you are

Marketing's catalogue problem (Module 02) was about predicting a *number*. Retention's problem is different: for each customer, will they come back or not? **Yes or no.** That is a **label**, not a quantity, and predicting labels is called **classification**. It is the most common kind of machine learning in business: fraud or fine, spam or not, will churn or will stay, which of five product categories.

In this module you build the churn model four different ways, each with a different idea of what "similar customers" means: nearest neighbours, yes/no questions, a probability curve, and counting. Then you learn the thing that separates professionals from beginners: **how to measure a classifier honestly**, because "90% accuracy" can mean a brilliant model or a useless one.

## What you will be able to do afterwards

- Recognise a classification problem and set up its features, target and baseline.
- Predict with k-nearest neighbours, computing the distances by hand.
- Grow a decision tree by hand with Gini impurity, then with scikit-learn.
- Turn a linear model into a probability with the sigmoid, and read logistic-regression coefficients.
- Build the confusion matrix and compute accuracy, precision, recall and F1 from it on paper.
- Draw an ROC curve, explain AUC, and choose a decision threshold from business costs.
- Handle more than two classes.
- Deliver a churn model for ShopWave and a survival model for the Titanic.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 03.1 | [Classification: predicting a label instead of a number](01-what-is-classification/) | 1 h |
| 03.2 | [K-nearest neighbours by hand](02-knn-by-hand/) | 1.5 h |
| 03.3 | [Decision tree classifier: Gini impurity by hand](03-decision-tree-classifier/) | 2 h |
| 03.4 | [Logistic regression: the sigmoid and probabilities](04-logistic-regression/) | 2 h |
| 03.5 | [Naive Bayes: counting your way to a prediction](05-naive-bayes/) *(optional)* | 1.5 h |
| 03.6 | [The confusion matrix: accuracy, precision, recall, F1](06-confusion-matrix-metrics/) | 2 h |
| 03.7 | [ROC curves, AUC and choosing a threshold by business cost](07-roc-auc-threshold/) | 1.5 h |
| 03.8 | [Multi-class classification](08-multiclass/) | 1 h |
| 03.P | [Mini-project: Titanic survival, then ShopWave churn](project/) | 3.5 h |

## The data in this module

- **By hand:** `data/shopwave_customers_small.csv`, 20 customers, target `churned` (9 churned, 11 stayed).
- **For real:** the ShopWave customer table built in lesson 01.9: features from orders before 1 September 2011, target = no purchase in September–November 2011. About 3,300 customers, 44% churned.

## The maths in this module

Four small pieces, each introduced where it is needed: the **distance** between two points (Pythagoras), a **purity score** for a group (Gini), the **sigmoid** curve that squashes any number into 0–1, and **ratios of counts** (precision, recall). Everything is computed by hand on a handful of rows before any library call.
