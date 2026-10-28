# 06.9 · Market basket analysis: support, confidence, lift

⏱ about 1.5 hours · Prerequisites: 01.6 (groupby, pivot), 03.5 (conditional probability) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

ShopWave's web team wants "frequently bought together" boxes on product pages, and the warehouse wants to shelve items that ship together next to each other. Both ask the same question: **which products end up in the same basket more often than chance would explain?**

There is no target to predict. You have 18,000 invoices and want patterns hiding in them. With 3,600 products there are over six million possible pairs, so looking by eye is out.

## 2. The idea

Think of a supermarket noticing that people who buy pasta often buy tomato sauce. Three questions make that precise:

1. **How common is the pair?** If only three baskets out of a million have both, who cares. (*support*)
2. **When someone buys pasta, how often do they also buy sauce?** (*confidence*)
3. **Is that more than you would expect anyway?** If 90% of all shoppers buy sauce, then 90% of pasta buyers doing so means nothing. (*lift*)

**Lift** is the key number: it compares "sauce among pasta buyers" with "sauce among everyone".

## 3. Definition

- **Basket (transaction)**: the set of items in one invoice.
- **Itemset**: a set of one or more items, e.g. {tea, cake}.
- **Association rule** A → B: "baskets with A tend to contain B". A is the *antecedent*, B the *consequent*.
- **Support** of an itemset: the share of baskets containing all its items.
- **Confidence** of A → B: the share of baskets with A that also contain B.
- **Lift** of A → B: confidence divided by B's overall support. Above 1: bought together more than by chance; 1: unrelated; below 1: less than by chance.
- **Apriori**: an algorithm that finds all itemsets above a minimum support, using the fact that a frequent itemset can only contain frequent smaller itemsets.

## 4. Formula

With $N$ baskets:

$$\text{support}(A) = \frac{\#\text{baskets containing } A}{N}$$

$$\text{confidence}(A \to B) = \frac{\text{support}(A \cup B)}{\text{support}(A)} = P(B \mid A)$$

$$\text{lift}(A \to B) = \frac{\text{confidence}(A \to B)}{\text{support}(B)} = \frac{\text{support}(A \cup B)}{\text{support}(A)\,\text{support}(B)}$$

Lift is symmetric (A → B and B → A have the same lift); confidence generally is not.

## 5. By hand

Eight baskets:

| basket | items |
|---|---|
| B1 | tea, cake, mug |
| B2 | tea, cake |
| B3 | tea, mug |
| B4 | cake, candle |
| B5 | tea, cake, card |
| B6 | candle, card |
| B7 | tea, cake, candle |
| B8 | mug, card |

**Single supports.** tea: B1, B2, B3, B5, B7 → 5/8 = **0.625**. cake: B1, B2, B4, B5, B7 → **0.625**. mug: B1, B3, B8 → 0.375. candle: B4, B6, B7 → 0.375. card: B5, B6, B8 → 0.375.

**tea → cake.** Both in B1, B2, B5, B7: support = 4/8 = **0.5**.
Confidence = 0.5 / 0.625 = **0.8**. Lift = 0.8 / 0.625 = **1.28**. Tea buyers are 28% more likely than average to buy cake.

**mug → tea.** Both in B1, B3: support 0.25. Confidence = 0.25 / 0.375 = **0.667**. Lift = 0.667 / 0.625 = **1.07**. Two-thirds sounds impressive, but tea is in 62.5% of all baskets anyway: barely any link.

**candle → card.** Both only in B6: support 0.125. Confidence = 0.125 / 0.375 = **0.333**. Lift = 0.333 / 0.375 = **0.89**. Below 1: candle buyers are *less* likely than average to buy a card.

**Apriori principle.** With a minimum support of 0.2, {candle, card} (0.125) cannot be frequent. Apriori knows this without counting: {card} is at 0.375, but any superset of an infrequent set is infrequent too, so once a pair fails, no triple containing it is ever examined.

## 6. In code

```python
from mlxtend.frequent_patterns import apriori, association_rules

basket = (sales.groupby(["Invoice", "Description"])["Quantity"].sum()
               .unstack(fill_value=0) > 0)          # invoices x products, True/False
frequent = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.0)
rules.sort_values("lift", ascending=False).head(10)
```

The notebook checks the eight-basket numbers above with both hand-written functions and mlxtend; all match.

## 7. On real data

**France, 2011.** 389 invoices, 1,543 different products, 20.6 products per basket on average. With a minimum support of 5%, Apriori finds 112 frequent itemsets and **86 rules** with lift above 1.

The top rules are **sets and colour variants**:

| if bought | then also | confidence | lift |
|---|---|---|---|
| skull paper plates | skull paper cups | 0.909 | 14.1 |
| children's cutlery, Dolly Girl | children's cutlery, Spaceboy | 0.893 | 12.9 |
| alarm clock, red | alarm clock, green | 0.838 | 8.6 |

Customers complete matching sets or buy several colours of the same item. Good uses: "complete the set" prompts, shelving set members together.

**The postage trap.** POSTAGE is on 77% of French invoices. Leave it in and you get **69 rules** of the form "X → POSTAGE" with confidence 96–100%. Their lift is about 1.29, close to the ceiling of 1 / 0.77 = 1.30. They are all true and all useless.

**The rare-pairs trap.** Lowering the minimum support from 5% to 3% to 2% takes the rule count from 86 to 1,186 to 6,078, and the top rules by lift come from pairs seen in as few as 8 baskets.

**Germany** (exercise 2) shows the same pattern: charlotte bags in two prints (lift 6.6), plasters in tins with different motifs, snack boxes in two designs.

**Stability** (exercise 3). Mining the first and second halves of 2011 separately, only **50 of 192** first-half rules reappear in the second half.

## 8. Where it breaks

- **Confidence alone misleads.** Popular items such as postage or carrier bags appear in high-confidence rules with everything. Sort by lift.
- **Rare items give spurious rules.** Low support plus high lift often means coincidence. Keep a sensible minimum support.
- **Rules are not causes.** "Plates → cups" does not mean a discount on plates will sell more cups; they may simply be bought for the same party. Test promotions with experiments.
- **Rules drift.** Seasonal ranges and one-off customers create rules that vanish next quarter. Check rules on a later period.
- **Combinatorial explosion.** Low minimum support on wide data can run out of memory. Filter to one country or category, or use FP-Growth (`mlxtend.frequent_patterns.fpgrowth`), which is faster.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: support, confidence and lift for cake → tea and card → mug. Is lift symmetric? Is confidence?
2. Repeat the analysis for Germany. Are the rules the same kind as in France?
3. Mine the first and second halves of 2011 separately. How many rules survive?

**Quiz**

1. What is the support of {A, B}?
2. Define confidence(A → B) in words and as a formula.
3. What does a lift of 1.28 mean? And 0.89?
4. State the Apriori principle.
5. Why is "anything → POSTAGE" a high-confidence but useless rule?

## 10. Recap

- Market basket analysis finds items bought together, without any target.
- **Support**: how common; **confidence**: P(B | A); **lift**: how much more than chance.
- **Apriori** prunes the search: infrequent sets cannot grow into frequent ones.
- On ShopWave, the strongest rules are matching sets and colour variants.
- Sort by lift, drop near-universal items, keep minimum support sensible, and check rules on fresh data.

**Next:** [06.10 Recommender systems](../10-recommenders/)
