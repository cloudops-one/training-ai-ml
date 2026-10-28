# 03.5 · Naive Bayes: counting your way to a prediction

⏱ about 1.5 hours · *optional* · Prerequisites: 03.1, fractions · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's support team receives 400 emails a day. They want the ones that are complaints routed to a senior agent within minutes, and everything else ("where is my order", "do you ship to Ireland") to the general queue.

A complaint has no fixed wording. But some words appear far more often in complaints ("refund", "broken", "disappointed", "never") and some almost never ("thanks", "wondering"). If you could *count* how often each word appears in past complaints versus past non-complaints, you could score a new email by multiplying those counts together.

That is Naive Bayes. It was the first spam filter, it is still a strong text baseline, it trains in milliseconds, and it works from counts you can do by hand.

## 2. The idea

Start with what you knew before reading the email: historically 40% of emails are complaints. That is your **prior**.

Now read the first word: "refund". Among past complaints, "refund" appeared in 75% of them; among non-complaints, in 10%. The word is 7.5× more typical of complaints. **Update** your belief: a lot more likely to be a complaint now.

Read the next word, update again. Repeat for every word. Naive Bayes is this updating, done with multiplication, with one simplification: it pretends the words are independent of each other given the class (that "refund" and "broken" appearing together is no more informative than each alone). That assumption is false, hence **naive**, and yet the method works remarkably well because it only needs to get the *ranking* right, not the exact probability.

## 3. Definition

**Bayes' theorem** says how to reverse a conditional probability:

$$P(\text{class} \mid \text{evidence}) = \frac{P(\text{evidence} \mid \text{class}) \cdot P(\text{class})}{P(\text{evidence})}$$

- $P(\text{class})$: the **prior**, how common the class is before seeing anything.
- $P(\text{evidence} \mid \text{class})$: the **likelihood**, how typical this evidence is for that class. This is what you count from training data.
- $P(\text{class} \mid \text{evidence})$: the **posterior**, what we want.
- $P(\text{evidence})$: the same for every class, so for *choosing* a class it can be ignored. It only matters to make the posteriors sum to 1.

**Naive Bayes classifier**: with several pieces of evidence $x_1, x_2, \dots$, assume they are independent given the class, so the likelihood becomes a product:

$$P(\text{class} \mid x_1, x_2, \dots) \propto P(\text{class}) \cdot P(x_1 \mid \text{class}) \cdot P(x_2 \mid \text{class}) \cdots$$

Compute this score for every class; predict the largest.

Three flavours, by the type of feature:

- **Bernoulli / Categorical NB**: yes/no or category features (did the email contain "refund"?).
- **Multinomial NB**: counts (how many times "refund" appeared). The text workhorse.
- **Gaussian NB**: continuous numbers (days since last order), modelled with a bell curve per class.

## 4. Formula

For the two-feature ShopWave case below, with class $C \in \{\text{churn}, \text{stay}\}$ and binary features $V$ (visited last month) and $E$ (opened an email):

$$\text{score}(C) = P(C) \times P(V = v \mid C) \times P(E = e \mid C)$$

$$P(\text{churn} \mid v, e) = \frac{\text{score}(\text{churn})}{\text{score}(\text{churn}) + \text{score}(\text{stay})}$$

Every term on the right is a **fraction counted from the training table**.

## 5. By hand

Ten past customers. $V$ = visited the site last month, $E$ = opened our last email, churned = left in the following quarter.

| # | visited (V) | opened email (E) | churned |
|---|---|---|---|
| 1 | yes | yes | no |
| 2 | yes | yes | no |
| 3 | yes | no | no |
| 4 | yes | yes | no |
| 5 | no | yes | no |
| 6 | yes | no | no |
| 7 | no | no | **yes** |
| 8 | no | no | **yes** |
| 9 | no | yes | **yes** |
| 10 | yes | no | **yes** |

**Priors.** 4 churned, 6 stayed: $P(\text{churn}) = 0.4$, $P(\text{stay}) = 0.6$.

**Likelihoods** (count within each class):

| | among churners (4) | among stayers (6) |
|---|---|---|
| $P(V = \text{no} \mid C)$ | rows 7, 8, 9 → 3/4 = 0.75 | row 5 → 1/6 ≈ 0.167 |
| $P(E = \text{no} \mid C)$ | rows 7, 8, 10 → 3/4 = 0.75 | rows 3, 6 → 2/6 ≈ 0.333 |

**Bayes' theorem on one feature first.** A new customer did *not* visit. Probability they churn?

$$P(\text{churn} \mid V = \text{no}) = \frac{P(V = \text{no} \mid \text{churn}) \, P(\text{churn})}{P(V = \text{no})} = \frac{0.75 \times 0.4}{4/10} = \frac{0.30}{0.40} = 0.75$$

Check by counting directly: the non-visitors are rows 5, 7, 8, 9; three of the four churned. 0.75. Bayes' theorem is just a rearrangement of counting.

**Naive Bayes on two features.** New customer: no visit, no email opened.

$$\text{score}(\text{churn}) = 0.4 \times 0.75 \times 0.75 = 0.225$$
$$\text{score}(\text{stay}) = 0.6 \times 0.167 \times 0.333 = 0.0333$$

$$P(\text{churn} \mid \text{no visit, no email}) = \frac{0.225}{0.225 + 0.0333} = \frac{0.225}{0.2583} \approx 0.87$$

**Predict churn**, with 87% confidence. Note the shape of the calculation: one prior, one fraction per feature, multiply, normalise. Nothing was fitted; everything was counted.

**The zero problem.** Suppose no stayer had ever opened zero emails: $P(E = \text{no} \mid \text{stay}) = 0/6 = 0$, and the stay score would be exactly 0 no matter what else was true. One unseen combination should not veto a class, so in practice every count gets +1 (**Laplace smoothing**, `alpha=1` in scikit-learn): $(2 + 1)/(6 + 2) = 0.375$ instead of $0.333$. The notebook shows both.

## 6. In code

```python
from sklearn.naive_bayes import CategoricalNB
nb = CategoricalNB(alpha=0)          # alpha=0: raw counts, matches the by-hand table
nb.fit(X, y)                         # X = [[visited, opened_email], ...] as 0/1
nb.predict_proba([[0, 0]])           # → [[0.13, 0.87]]
```

Then `GaussianNB` on the 20-customer table (continuous features), and `MultinomialNB` on a tiny set of support emails, where you print the per-word likelihoods and see exactly which words pushed the decision.

## 7. On real data

Gaussian NB on the ShopWave customer table (orders, spend, days since last order…) versus the logistic regression from 03.4, scored with the 03.6 metrics. Its churn F1 looks fine; its full report does not: it flags almost everyone as a churner, because the Gaussian assumption fits skewed spend data badly. A useful lesson in reading the whole report, and a log transform (exercise 4) repairs much of it. Then the lesson's real home: `MultinomialNB` on 2,000 short product reviews (positive vs negative; generated with a seed so the notebook needs no download, real review datasets arrive in Module 09), where it is fast and accurate, and you can list the ten words that most strongly signal each class.

## 8. Where it breaks

- **Features that are not independent.** "refund" and "money back" say the same thing; NB counts the evidence twice and becomes over-confident. Its *ranking* usually survives; its *probabilities* do not. Do not read NB's 0.99 as a calibrated 99%.
- **Continuous features that are not bell-shaped.** Gaussian NB assumes a normal curve per class. Spend is heavily skewed; take a log first or use another model.
- **Unseen values** make a likelihood zero. Always smooth (`alpha ≥ 1`).
- **Correlated numeric features on tabular data**: logistic regression, trees and ensembles (Module 05) almost always beat NB there. NB's territory is text, high-dimensional sparse counts, and tiny training sets.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: a new customer visited (V = yes) and did *not* open the email. Compute both scores and the churn probability.
2. By hand: redo exercise 1 with Laplace smoothing (+1 to every count, +2 to every class total). How much did the probability move?
3. In code: build the per-word likelihood table for the five-email example and explain, in words, why the test email was classified as it was.
4. In code: on the real customer table, apply `np.log1p` to `spend`, `orders` and `products` before `GaussianNB`. Compare accuracy, macro F1 and the recall of both classes. What changed, and why?

**Quiz**

1. In Bayes' theorem, which term is the "prior" and what does it represent?
2. Why is the method called *naive*?
3. What does Laplace smoothing prevent?
4. Which NB variant fits word counts, and which fits continuous measurements?
5. Give one reason NB is popular for text even though its independence assumption is false.

## 10. Recap

- Bayes' theorem: $P(\text{class} \mid \text{evidence}) \propto P(\text{evidence} \mid \text{class}) \, P(\text{class})$, and both factors are counted from data.
- Naive Bayes multiplies one likelihood per feature, assuming features are independent given the class.
- By hand: prior × fraction × fraction, for each class; normalise; predict the larger. 0.87 churn for "no visit, no email".
- Smooth counts (+1) so a never-seen value cannot zero out a class.
- Strong and fast for text and sparse counts; usually beaten on tabular numeric data.

**Next:** [03.6 The confusion matrix](../06-confusion-matrix-metrics/): now that you have several classifiers, learn to score them honestly.
