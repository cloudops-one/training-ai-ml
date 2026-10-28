# 00.1 · Why machine learning? Rules vs learning from data

⏱ about 1 hour · Prerequisites: Python basics · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Your first task at ShopWave is small. Support staff keep missing fraudulent orders: someone buys £900 of gifts with a stolen card, the goods ship, and ShopWave eats the loss. The head of operations asks: *"Can you write something that flags suspicious orders before they ship?"*

You know Python. So you do what any programmer does: you write rules.

```python
def is_suspicious(order):
    if order.total > 500 and order.customer_age_days < 7:
        return True
    if order.country != order.card_country:
        return True
    if order.quantity > 20 and order.is_gift_wrapped:
        return True
    return False
```

It works for a week. Then:

- A loyal customer who moved to Spain gets flagged on every order and complains.
- Fraudsters learn to keep orders under £500.
- Operations sends you a list of twelve more patterns they have noticed.
- The rules now contradict each other and nobody, including you, can predict what the function will say.

This is not a bug in your code. It is the limit of the *approach*. Some problems cannot be solved by writing down the rules, because **nobody knows the rules**. We only have examples.

## 2. The idea

Think about how you learned to recognise your friends' handwriting. Nobody gave you rules ("Priya's *g* has a closed loop and slants 15° left"). You saw many examples and your brain found the pattern itself.

Machine learning is the same move applied to a program. Instead of telling the computer *how* to decide, you show it many examples of *inputs and the right answer*, and let it find the pattern.

```
Traditional programming:    rules + data  →  answers
Machine learning:           data + answers  →  rules
```

That is the whole idea. Everything else in this course is detail about *how* the computer finds the rules, how to check it found good ones, and how to use them.

## 3. Definition

**Machine learning** is building programs that improve at a task by learning patterns from data, rather than by being given hand-written rules.

A tidy version from Tom Mitchell (1997): a program learns if its **performance** at a **task** improves with **experience**. For ShopWave fraud:

- **Task**: decide whether an order is fraudulent.
- **Experience**: 50,000 past orders, each marked "fraud" or "fine" by what happened afterwards.
- **Performance**: fraction of frauds caught, and fraction of honest customers wrongly flagged.

The learned rules are called a **model**. The columns you give it (`total`, `customer_age_days`, `country`, …) are **features**. The thing it is trying to predict (`fraud` yes/no) is the **target**.

## 4. Formula

There is one formula for this lesson and it is not maths yet, it is the shape of every ML problem:

$$\hat{y} = f(X)$$

- $X$: the features of one order (a row of numbers).
- $f$: the model, a function with adjustable parameters.
- $\hat{y}$ ("y-hat"): the model's prediction.
- $y$ (no hat): the true answer, known for past orders.

**Learning** means: adjust the parameters of $f$ so that $\hat{y}$ is as close to $y$ as possible across all the past examples. In Module 02 you will do this adjustment by hand. For now, just hold the shape in your head: *inputs go in, a guess comes out, we tune the function until the guesses are good*.

## 5. By hand

Here are eight past ShopWave orders. Cover the last column and try to write a rule that separates fraud from fine.

| order | total (£) | account age (days) | items | country matches card | fraud? |
|---|---|---|---|---|---|
| 1 | 45 | 400 | 2 | yes | no |
| 2 | 620 | 3 | 9 | no | **yes** |
| 3 | 310 | 120 | 4 | yes | no |
| 4 | 880 | 1 | 12 | yes | **yes** |
| 5 | 95 | 2 | 1 | yes | no |
| 6 | 540 | 800 | 6 | no | no |
| 7 | 430 | 5 | 7 | no | **yes** |
| 8 | 70 | 30 | 3 | yes | no |

Things you might notice:

- *"Total over 400"* catches orders 2, 4, 7 (all fraud) but also 6 (fine). 3 of 4 right.
- *"Account under 7 days"* catches 2, 4, 5, 7: three frauds and one honest new customer. 3 of 4 right.
- *"Total over 400 **and** account under 7 days"* catches exactly 2, 4, 7. 3 of 3, no false alarms.

With eight rows you can do this by eye. With 50,000 rows and 40 columns you cannot, and the best rule may involve combinations no human would think to try. **A learning algorithm does this search for you**: it tries thresholds, combinations and weightings, scores each one against the known answers, and keeps the best. That is all a decision tree (Module 03) does, just very fast and very thoroughly.

Write down your best rule now. You will compare it with what a model finds in the notebook.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). You will:

1. Load the same eight orders, then a few hundred more generated the same way.
2. Write the hand rule as a Python function and score it.
3. Train a tiny decision tree with three lines of scikit-learn and score it.
4. Print the rules the tree found and compare them with yours.

You are not expected to understand the scikit-learn code yet. The point is to *see* that the computer discovered rules from examples, and that they are readable.

```python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)               # learn from examples
print(export_text(model, feature_names=list(X.columns)))
```

## 7. On real data

The notebook ends with a glimpse of the real ShopWave data (one million rows of actual online-retail transactions). You will look at it only to appreciate the scale: why rules by hand are hopeless and why Module 01 teaches you to handle data of this size before anything else.

## 8. Where it breaks

Machine learning is not always the answer. Use **rules** when:

- The rule is known and fixed. VAT is 20%; do not train a model to learn that.
- You need a guarantee. "Never ship to a sanctioned country" must be a hard rule, not a probability.
- You have no examples. A model learns from the past; a brand-new product line has no past.
- A wrong answer is unacceptable and unexplainable. Some regulated decisions need rules you can print.

Use **machine learning** when:

- Nobody can write the rules, but you have many examples with answers (fraud, churn, demand).
- The pattern changes and you would rather retrain than rewrite (spam, recommendations).
- The input is unstructured: images, text, audio. There is no `if` statement for "this photo contains a mug".

The real skill, which you build across this course, is knowing which one you are looking at.

## 9. Check yourself

**Exercises** (in [exercises.ipynb](exercises.ipynb))

1. Add two more orders to the table that your hand rule gets wrong. What would you need to change?
2. Change `max_depth=2` to `max_depth=5` in the notebook. Does the tree's accuracy go up? Do its rules still make sense? (This is your first sighting of overfitting.)
3. Think of three tasks in a business you know. For each, say: rules or ML, and why.

**Quiz** (answers at the bottom of the notebook)

1. In one sentence, what does "learning" mean for a program?
2. A model predicts a house price from its size. Which is the feature and which is the target?
3. True or false: machine learning always beats hand-written rules when enough data is available.
4. Why did the fraud rules become unmaintainable?
5. Name one ShopWave problem that should stay rule-based.

## 10. Recap

- Traditional code is *rules + data → answers*. Machine learning is *data + answers → rules*.
- A **model** is a function $f$ with parameters; **training** tunes the parameters so predictions $\hat{y}$ match known answers $y$.
- **Features** are the inputs; the **target** is what you predict.
- ML wins when nobody can write the rules but examples exist; rules win when the rule is known, must be guaranteed, or there is no data.
- A decision tree is just an algorithm that searches for the best if/else rules, which is what you did by hand on eight rows.

**Next:** [00.2 The AI family tree](../02-ai-family-tree/) places machine learning next to AI, deep learning and generative AI so you can tell which is which.
