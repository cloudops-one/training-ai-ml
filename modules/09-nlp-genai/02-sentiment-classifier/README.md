# 09.2 · Sentiment classifier on product reviews

⏱ about 1.5 hours · Prerequisites: 09.1, 03.4 (logistic regression), 03.6 (confusion matrix) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's support team wants every unhappy review on their screen within an hour, so they can contact the customer before the complaint spreads. With 50,000 reviews a year, that is about 140 a day. Nobody reads 140 reviews a day carefully.

You have 10,000 reviews already labelled positive or negative. That is a **supervised classification** problem, exactly like churn in Module 03, except the features are words. 09.1 turned the words into TF-IDF numbers. This lesson puts them into logistic regression and asks how well counting words can read a mood, and where it fails.

## 2. The idea

Give every word a vote. *great* votes strongly for positive, *waste* strongly for negative, *mug* hardly votes at all. Add up the votes in a review, squash the total into a probability with the sigmoid, and you have a sentiment score.

Nobody writes the votes by hand. **Logistic regression learns them** from the labelled reviews, one coefficient per word, exactly as it learned one coefficient per customer feature in 03.4. A model with 66,583 columns sounds frightening, but it is the same model you already know.

## 3. Definition

- **Text classifier**: a vectoriser (text → numbers) followed by a classifier (numbers → label), usually chained in one `Pipeline` so the vocabulary is learned from training data only (Module 04).
- **Coefficient of a word**: its vote. Positive pushes towards *positive*, negative towards *negative*. Because TF-IDF scales are comparable, the size of a coefficient shows how strongly a word signals sentiment.
- **Negation**: a word like *not* that flips the meaning of the next words. Bag-of-words cannot see it unless it is captured in a bigram.

## 4. Formula

The same logistic regression as 03.4, with words as features:

$$P(\text{positive} \mid \text{review}) = \sigma\Big(b + \sum_{t} w_t \, x_t\Big) \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

where $x_t$ is the TF-IDF value of word $t$ in the review and $w_t$ its learned coefficient. Words missing from the review have $x_t = 0$ and contribute nothing.

## 5. By hand

A toy model that uses plain counts and four words: *great* $+2$, *good* $+1$, *not* $-2$, *bad* $-2$, intercept $b = 0$. Every other word has weight 0.

| review | sum of votes $z$ | $\sigma(z)$ | verdict |
|---|---|---|---|
| "great mug, great price" | $2 + 2 = 4$ | 0.982 | positive ✓ |
| "great mug but not great" | $2 - 2 + 2 = 2$ | 0.881 | positive ✗ |
| "not good" | $-2 + 1 = -1$ | 0.269 | negative ✓ |
| "not bad" | $-2 - 2 = -4$ | 0.018 | negative ✗ |

The two mistakes are the whole story of bag-of-words sentiment. *"not great"* counts as *great*, and *"not bad"* counts as two negative words. The model gets *"not good"* right only by luck: *not* happens to point the same way as the true meaning.

Bigrams give *not good* and *not bad* their own columns, with their own learned votes. 09.1 showed the price: the vocabulary grows from 14,060 to 66,583 columns.

## 6. In code

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                      LogisticRegression(max_iter=1000))
model.fit(train["text"], train["label"])
model.predict_proba(["The mug arrived broken."])[:, 1]   # probability of positive
```

The pipeline takes raw strings. It learns the vocabulary from the training reviews only, so no test-set word leaks into training.

## 7. On real data

8,000 training reviews, 2,000 test reviews, half positive. Guessing the majority class scores **50%**.

| features | test accuracy | test AUC |
|---|---|---|
| single words | 87.5% | 0.941 |
| single words, stop words removed | 86.0% | 0.928 |
| **single words + word pairs** | **88.4%** | **0.949** |

Removing stop words *costs* 1.6 points, because it deletes *not* and *no*. Word pairs add almost a point.

**The learned votes.** The strongest negative word is ***not*** (−7.66), stronger than *poor*, *bad*, *waste* or *worst*. The strongest positive word is *great* (+7.86), then *love*, *best*, *excellent*. The bigram *not good* gets its own vote of −1.47, and *not bad* gets +0.49.

**The test confusion matrix** (rows = truth, columns = prediction):

| | predicted negative | predicted positive |
|---|---|---|
| **negative** | 898 | 102 |
| **positive** | 130 | 870 |

Precision for negative reviews is 0.874 and recall 0.898: of every 10 reviews the model sends to support, about 9 really are unhappy, and it catches 9 in 10 unhappy reviews.

**Where it still fails** (the notebook prints probabilities of *positive*):

- *"This is not good."* → 0.15. Correct.
- *"Not bad at all, I'm pleasantly surprised."* → **0.18**. Wrong. *not* (−7.66) and *bad* (−3.72) outvote the small *not bad* bigram.
- *"Oh great, another charger that died after a week. Just what I needed."* → 0.53. Sarcasm: *great* pulls up, *died* pulls down, the model shrugs.
- *"Works."* → 0.92. One positive word is enough.

232 of the 2,000 test reviews are misclassified. Read the most confident mistakes in the notebook: several are mixed reviews (*"not great though"*), and some of the "wrong" predictions are actually right. *"Great easy to use … it looks easy to use"* is labelled negative in the data. **Labels are noisy**, and a model can never score 100% against noisy labels.

## 8. Where it breaks

- **Negation and contrast.** *not bad*, *I expected to hate it but…*: word order matters and bag-of-words only sees a little of it through bigrams.
- **Sarcasm and irony.** Positive words used to mean the opposite.
- **Domain shift.** These reviews are mostly books, music and electronics. A model trained here has never seen *chipped* or *retrospot*; a ShopWave mug review may score oddly. Check it on your own data before trusting it.
- **Label noise.** Star ratings are not perfect labels. Some 4-star reviews complain; some 2-star reviews praise.
- **Coefficients are not causes.** *warranty* gets a negative vote because people mention it when something broke, not because the word is bad.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand, with the toy votes from section 5: score *"good good bad"* and *"not not bad"*. Then add a bigram *not bad* with vote $+5$ and rescore *"not bad"*.
2. Train the bigram model with `min_df` of 1, 2, 5 and 20. How do vocabulary size and test accuracy change?
3. Write five reviews of ShopWave products that you think will fool the model. How many do?
4. Learning curve: train the bigram model on the first 100, 500, 2,000 and 8,000 training reviews and plot test accuracy. Keep the numbers; 09.3 compares them with embeddings.

**Quiz**

1. What does one coefficient in this model mean?
2. Why did removing stop words lower accuracy?
3. Why does *"Not bad at all"* still come out negative even with bigrams?
4. Support wants to miss as few unhappy reviews as possible. Which metric do they care about, and how would you change the model's behaviour without retraining?
5. Name two reasons the model cannot reach 100% accuracy on this data.

## 10. Recap

- A text classifier is a **vectoriser + classifier** in one pipeline; TF-IDF + logistic regression is the classic, strong baseline.
- Each word gets a learned **vote** (coefficient). On the reviews: *great* +7.86, *not* −7.66.
- Results on 2,000 test reviews: 88.4% accuracy, AUC 0.949, against a 50% baseline. Removing stop words hurts (86.0%).
- It fails on **negation, sarcasm, mixed reviews and domain shift**, and labels themselves are noisy.
- The threshold and metric choices from Module 03 apply unchanged: decide what a missed complaint costs.

**Next:** [09.3 Embeddings](../03-embeddings/) give words and sentences a meaning, not just a count.
