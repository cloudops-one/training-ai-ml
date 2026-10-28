# 05.1 · Ensembles: why many weak models beat one strong one

⏱ about 1 hour · Prerequisites: 03.2–03.4, 03.7 (AUC), 04.7 (cross-validation) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You have three churn models from Module 03: logistic regression, k-nearest neighbours and a shallow decision tree. Their cross-validated AUCs are close (0.74, 0.73, 0.73). The head of data asks which one to ship.

A reasonable instinct is to pick the best and throw the others away. But each model gets *different* customers wrong. KNN is fooled by customers who look like their neighbours but behave differently; the tree is fooled by customers just on the wrong side of a threshold; logistic regression is fooled by anything that is not a straight-line effect. What if, instead of choosing, you let them **vote**?

## 2. The idea

At a village fair in 1906, Francis Galton watched 787 people guess the weight of an ox. Almost nobody guessed right. The **median** guess was within 1% of the true weight. Individual errors, some too high and some too low, cancelled out.

That is the whole idea of an **ensemble**: combine many models so that their mistakes cancel. It works on one condition, and it is the important one: **the mistakes must be different.** Ask the same person 787 times and you get the same wrong answer 787 times. Ask 787 different people and the errors average away.

```
one model:            ✓ ✗ ✓ ✓ ✗ ✓ ✗ ✓ ✓ ✗     60% right

five different models, one row each:
   row 1:  ✓ ✓ ✗ ✓ ✗   → majority ✓
   row 2:  ✗ ✓ ✗ ✗ ✓   → majority ✗
   row 3:  ✓ ✗ ✓ ✓ ✓   → majority ✓
   ...                     majority is right 68% of the time
```

## 3. Definition

An **ensemble** is a model made of several models whose predictions are combined.

- **Hard voting**: each model predicts a class; the majority wins.
- **Soft voting**: each model gives a probability; the ensemble averages them. Usually better, because a confident model counts for more than an unsure one.
- **Bagging** (*bootstrap aggregating*): train *the same kind* of model many times on random resamples of the data, then average. Different data → different mistakes. Random forest is bagging with trees. *(05.2)*
- **Boosting**: train models *one after another*, each focusing on what the previous ones got wrong, then add them up. Gradient boosting, XGBoost and LightGBM are boosting. *(05.3, 05.4)*
- **Stacking**: train several different models, then train a *final* model that learns how much to trust each one. Powerful, easy to leak; mentioned here, used rarely.

| | Models trained | Each model sees | Combined by | Mainly reduces |
|---|---|---|---|---|
| Voting | different kinds, independently | all training data | vote / average | errors specific to one method |
| Bagging | same kind, independently, in parallel | a random resample | average | **variance** (wobbliness) |
| Boosting | same kind, one after another | all data, reweighted towards mistakes | weighted sum | **bias** (systematic error) |
| Stacking | different kinds | all data | a second model | both, if done carefully |

## 4. Formula

Suppose each of $n$ voters is right with probability $p$, independently of the others. The majority is right when more than half of them are right. The number right follows a **binomial** distribution, so

$$P(\text{majority right}) = \sum_{k > n/2}^{n} \binom{n}{k}\, p^{k}\,(1-p)^{\,n-k}$$

- $\binom{n}{k}$, "n choose k": the number of ways to pick which $k$ of the $n$ voters are right. $\binom{5}{3} = 10$.
- $p^k$: the chance those $k$ are all right.
- $(1-p)^{n-k}$: the chance the other $n-k$ are all wrong.

If $p > 0.5$ and the voters are independent, this rises towards 1 as $n$ grows. If the voters are copies of each other, it stays at $p$ no matter how many you add.

## 5. By hand

Five independent voters, each right 60% of the time ($p = 0.6$). The majority is right when 3, 4 or 5 of them are right.

| right ($k$) | $\binom{5}{k}$ | $0.6^k$ | $0.4^{5-k}$ | product |
|---|---|---|---|---|
| 3 | 10 | 0.216 | 0.16 | 0.3456 |
| 4 | 5 | 0.1296 | 0.4 | 0.2592 |
| 5 | 1 | 0.07776 | 1 | 0.07776 |
| | | | **sum** | **0.6826** |

Five mediocre voters, 60% each → a 68% committee. The notebook carries the same sum on:

| voters | majority right |
|---|---|
| 1 | 60.0% |
| 5 | 68.3% |
| 25 | 84.6% |
| 101 | 97.9% |

Then it simulates 10,000 rounds of 25 random voters and gets 84.9%, confirming the formula.

**The catch.** Real models trained on the same data are *not* independent; they tend to fail on the same hard customers. So real ensembles gain much less than this table promises. The whole craft of ensembles (random resamples, random feature subsets, sequential correction) is about making the members **as different as possible** while each stays reasonably good.

## 6. In code

```python
from sklearn.ensemble import VotingClassifier

vote = VotingClassifier(
    estimators=[("logreg", logreg), ("knn", knn), ("tree", tree)],
    voting="soft",                 # average predicted probabilities
)
cross_val_score(vote, X_train, y_train, cv=cv, scoring="roc_auc").mean()
```

`VotingClassifier` is itself a model: it has `fit`, `predict` and `predict_proba`, and it can go inside a pipeline or a cross-validation like any other.

## 7. On real data

The notebook scores the three Module 03 models and their soft vote on the ShopWave churn table (5-fold CV on the 2,487 training customers, then the 830 test customers):

| model | CV AUC | test AUC |
|---|---|---|
| logistic regression (scaled) | 0.740 | 0.741 |
| KNN, k = 51 (scaled) | 0.732 | 0.734 |
| decision tree, depth 3 | 0.731 | 0.724 |
| **soft vote of all three** | **0.744** | 0.741 |

The vote is the best in cross-validation, by a hair, and ties on test. That is typical when the members are all decent and mostly agree: the ensemble is *at least as good as the best member* and more stable, but it does not work miracles. The notebook also measures how often the three models disagree: they give the same answer for 82% of test customers, so voting can only change the outcome for the other 18%. That is why the gain is small.

## 8. Where it breaks

- **Correlated members.** Three models that make the same mistakes vote for the same mistakes. Diversity matters more than individual strength.
- **One strong, two weak.** In hard voting, two bad models can outvote a good one. Soft voting, or weights (`weights=[2, 1, 1]`), helps.
- **Cost.** Three models means three times the training, prediction time and maintenance.
- **Explainability.** "The average of a logistic regression, a KNN and a tree" is harder to explain to a CEO than one model. Lesson 05.8 fixes this for any model with SHAP.
- **Stacking leaks easily.** If the final model is trained on predictions the members made on their *own* training rows, it learns to trust overfit members. scikit-learn's `StackingClassifier` uses cross-validated predictions to avoid this; hand-rolled stacking often does not.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: three independent voters, each 70% right. What is the chance the majority is right? Check with code.
2. Write a function `majority(n, p)` and plot majority accuracy against $n$ (1 to 51, odd only) for $p$ = 0.55, 0.6 and 0.45. What happens when $p < 0.5$?
3. Simulate *correlated* voters: 25 voters who each copy a shared "mood" coin with probability 0.8 and otherwise flip their own 60% coin. How much does majority accuracy drop compared with 25 independent voters?
4. On the churn table, compare hard and soft voting, and try `weights=[2, 1, 1]`. Which is best in CV?

**Quiz** (answers at the bottom of the notebook)

1. In one sentence, why does Galton's crowd beat almost every individual?
2. What single condition must hold for an ensemble to beat its members?
3. Bagging trains models in ______; boosting trains them in ______.
4. Soft voting averages ______; hard voting counts ______.
5. Five voters each 60% right give a 68% majority. Why is a real ensemble of five churn models unlikely to gain that much?

## 10. Recap

- An **ensemble** combines many models so their mistakes cancel.
- With independent voters each better than chance, majority accuracy rises with the number of voters: 60% → 68% (5) → 85% (25), by the binomial sum.
- Real models are correlated, so the gain is smaller; **diversity** is what ensembles are engineered for.
- **Voting** combines different models; **bagging** averages resampled copies (reduces variance); **boosting** chains models that fix each other's errors (reduces bias); **stacking** learns how to combine.
- On ShopWave churn, a soft vote edges the best single model in CV (0.744 vs 0.740) and ties on test.

**Next:** [05.2 Bagging and Random Forest](../02-random-forest/) turns one decision tree into hundreds of different ones.
