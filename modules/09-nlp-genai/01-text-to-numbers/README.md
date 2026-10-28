# 09.1 · Text to numbers: cleaning, tokens, bag-of-words, TF-IDF by hand

⏱ about 2 hours · Prerequisites: 01.3–01.5 (Pandas), 03.4 (logistic regression) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's head of customer experience has 10,000 product reviews in a spreadsheet and one question: *"Which ones are angry, and what are they angry about?"*

Every model you have built needs numbers: `visits`, `spend`, `days_since_last`. A review is a string like *"The mug arrived broken and nobody replied."* You cannot multiply it by a coefficient or measure its distance to another review. Before any machine learning can happen, text has to become a row of numbers.

This lesson is that conversion. It is old, simple and still used everywhere, because it is fast, needs no GPU and is easy to explain.

## 2. The idea

Tip a review out onto the table like a bag of Scrabble tiles and count the words. Throw away the order, keep the counts. *"great mug, great price"* becomes `great: 2, mug: 1, price: 1`. That is a **bag of words**.

Do it for every review using the same list of words (the **vocabulary**) and you get a table: one row per review, one column per word, counts in the cells. Now it is just a very wide spreadsheet, and every tool from Modules 02–06 works on it.

One refinement makes it much better. A word like *"the"* appears in nearly every review, so seeing it tells you nothing. A word like *"broken"* appears in a few, so seeing it tells you a lot. **TF-IDF** turns counts into scores that reward words that are frequent in *this* review but rare across *all* reviews.

## 3. Definition

- **Token**: one unit of text after splitting, usually a word. **Tokenising** is the splitting.
- **Vocabulary**: every distinct token the model knows, fixed when you fit.
- **Bag-of-words / document-term matrix**: rows are documents (reviews), columns are vocabulary words, cells are counts.
- **Stop words**: very common words (*the, and, is*) that are sometimes removed.
- **Stemming**: chopping word endings by rule (*arrived, arriving* → *arriv*). **Lemmatisation**: mapping to the dictionary form using grammar (*broken* → *break*, *mice* → *mouse*).
- **n-gram**: a run of *n* consecutive tokens. Bigrams of *"not good at all"* are *not good, good at, at all*.
- **TF (term frequency)**: how often a word appears in one document.
- **IDF (inverse document frequency)**: how rare a word is across all documents.

## 4. Formula

For word $t$ in document $d$, with $N$ documents in total and $\text{df}(t)$ the number of documents containing $t$:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

The textbook IDF is $\ln\big(N / \text{df}(t)\big)$. scikit-learn uses a smoothed version, so that no word gets a weight of exactly zero and nothing divides by zero:

$$\text{idf}(t) = \ln\!\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$

Finally each review's row is divided by its length (the square root of the sum of squares, called **L2 normalisation**), so a long review and a short one with the same word mix get the same scores.

## 5. By hand

Four tiny reviews:

| | review |
|---|---|
| R1 | great mug great price |
| R2 | mug arrived broken |
| R3 | great lights |
| R4 | lights arrived late |

**Bag of words.** The vocabulary, sorted: *arrived, broken, great, late, lights, mug, price* (7 words).

| | arrived | broken | great | late | lights | mug | price |
|---|---|---|---|---|---|---|---|
| R1 | 0 | 0 | **2** | 0 | 0 | 1 | 1 |
| R2 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| R3 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| R4 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |

**IDF for two words.** $N = 4$.

- *great* appears in 2 reviews: $\text{idf} = \ln(5/3) + 1 = 0.5108 + 1 = \mathbf{1.5108}$
- *price* appears in 1 review: $\text{idf} = \ln(5/2) + 1 = 0.9163 + 1 = \mathbf{1.9163}$

The rarer word gets the bigger weight. *mug*, *arrived* and *lights* are in 2 reviews each, so they also get 1.5108; *broken* and *late* get 1.9163.

**TF-IDF for R1.**

| word | tf | × idf | = raw |
|---|---|---|---|
| great | 2 | 1.5108 | 3.0217 |
| mug | 1 | 1.5108 | 1.5108 |
| price | 1 | 1.9163 | 1.9163 |

Length: $\sqrt{3.0217^2 + 1.5108^2 + 1.9163^2} = \sqrt{9.1306 + 2.2826 + 3.6722} = \sqrt{15.0854} = 3.884$.

Divide each by 3.884: **great 0.778, mug 0.389, price 0.493.** *great* still leads because it appears twice, but *price* has overtaken *mug* because it is the rarer word. scikit-learn's `TfidfVectorizer` gives exactly these numbers.

## 6. In code

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

counts = CountVectorizer().fit_transform(docs)      # bag of words
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)                        # TF-IDF, rows L2-normalised
tfidf.get_feature_names_out(), tfidf.idf_            # vocabulary and IDF weights

TfidfVectorizer(ngram_range=(1, 2), min_df=2)        # words + word pairs, drop one-offs
```

`X` is a **sparse matrix**: it stores only the non-zero cells, because almost every cell is zero.

## 7. On real data

The notebook loads 10,000 Amazon product reviews standing in for ShopWave's (see the [module README](../README.md#data)):

- 746,200 tokens in total, 34,590 different words. **16,853 words (49%) appear exactly once.** Language has a very long tail.
- The 15 most common words (*the, and, to, it, of, this, is …*) make up **26.9%** of all tokens and say nothing about sentiment.
- With stop words removed, the top words become *book, great, good, like, just, read, time, movie …*: what people bought and how they felt.
- On the 8,000 training reviews, the vocabulary has 30,631 words, but a typical review uses about 55 of them: **0.18% of the cells are non-zero**. Keeping only words seen in at least 2 reviews (`min_df=2`) halves the vocabulary to 14,060. Adding word pairs (`ngram_range=(1, 2)`) grows it to 66,583.

09.2 trains a classifier on exactly these features.

## 8. Where it breaks

- **Word order is gone.** *"Not good, actually great"* and *"not great, actually good"* have the same bag.
- **Stop-word lists can delete meaning.** scikit-learn's English list has 318 words and includes *not* and *no*. Remove stop words before a sentiment model and *"not good"* becomes *"good"*. 09.2 measures the damage.
- **No notion of meaning.** *mug* and *cup* are as different as *mug* and *bicycle*. Embeddings (09.3) fix this.
- **The default tokeniser drops one-letter words** (*I*, *a*) and splits on apostrophes (*don't* → *don*). Usually fine, occasionally surprising.
- **Unknown words are ignored.** A word that was not in the training vocabulary gets no column. New product names are invisible until you refit.
- **Very wide and sparse.** Tens of thousands of columns. Linear models cope well; trees and KNN do not.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: build the bag-of-words table for *"lovely lights"*, *"lights broken"*, *"lovely lovely mug"*. Then compute the TF-IDF of *lovely* in the third review. Check in code.
2. List the bigrams of *"the mug was not good at all"*. Which bigram carries the sentiment?
3. Fit `CountVectorizer(stop_words="english")` on *"this is not good"* and print the vocabulary. What happened?
4. On the 10,000 reviews, find the 10 words whose share of appearances in negative reviews is highest (among words used at least 50 times).

**Quiz** (answers at the end of the notebook)

1. What does a bag-of-words row throw away?
2. Why does *the* get a low IDF?
3. In TF-IDF, why divide each row by its length?
4. Why can removing stop words hurt a sentiment model?
5. Your vocabulary has 30,000 words but a review uses 50. What kind of matrix stores this efficiently?

## 10. Recap

- Models need numbers. **Bag-of-words** counts each vocabulary word in each document; order is lost.
- **TF-IDF** = term frequency × inverse document frequency, then each row is scaled to length 1. Rare words that appear in this document score highest.
- By hand on four reviews: IDF 1.5108 for a word in 2 of 4 reviews, 1.9163 for a word in 1; R1 becomes great 0.778, mug 0.389, price 0.493.
- Real text is a long tail: half the words appear once, and the top 15 words are a quarter of all tokens.
- Watch the cleaning: stop-word lists contain *not*.

**Next:** [09.2 Sentiment classifier](../02-sentiment-classifier/) puts these features into logistic regression and asks how well counting words can read a mood.
