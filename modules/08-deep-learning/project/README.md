# 08.P · Mini-project: ShopWave product image classifier

⏱ about 3.5 hours · Prerequisites: all of Module 08 (08.11 optional) · Notebook: [project.ipynb](project.ipynb) · Solution: [solution.ipynb](solution.ipynb)

---

## The brief

ShopWave's new "garden & flowers" range launches next month. Suppliers have uploaded **3,670 photos** of five kinds of flower: daisy, dandelion, rose, sunflower and tulip. Every photo needs the right category before it can go on the website, and the tagging team cannot keep up.

The head of e-commerce wants to know:

1. **Can a model tag these photos well enough to help?**
2. **Where does it go wrong, and how often?**
3. **Which photos can it tag on its own, and which should a person check?**
4. **Is it ready to go live, and if not, what is missing?**

Your job is to build the tagger, test it honestly, and answer those four questions in plain language.

## The data

The 5-class flower photo set published by the TensorFlow team (originally from Flickr, Creative Commons licences). The notebook downloads it once (229 MB) and splits each class 70% / 15% / 15% into training, validation and test sets: 2,568 / 551 / 551 photos. Photos are resized to 160 × 160 and kept as `uint8` (282 MB in memory for all three sets), which fits on an 8 GB laptop.

The loading code is provided in the first cell so you can spend your time on the modelling.

## What to do

Work through the `TODO` cells in [project.ipynb](project.ipynb):

1. **Look at the data.** Photos per class, a few examples, and the baseline accuracy of always predicting the largest class.
2. **Build the model.** A frozen pretrained base (EfficientNetB0 or MobileNetV3Small), data augmentation, dropout and a new 5-class output layer. Report trainable and frozen weights.
3. **Train with early stopping.** At most 15 epochs, `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`. Plot training and validation accuracy.
4. **Evaluate on the test set, once.** Accuracy, per-class precision and recall, confusion matrix.
5. **Error analysis.** Show at least 10 misclassified photos and sort them into groups.
6. **A confidence rule.** For thresholds from 0.5 to 0.95: what share of photos would the model tag alone, and how accurate is it on those? Recommend a threshold.
7. **Save, reload, check.** Save to the repo's `models/` folder (git-ignored), load it back, confirm identical predictions.
8. **Ready to deploy?** One paragraph for the head of e-commerce.

**Time.** Training EfficientNetB0 for 15 epochs on 2,568 photos took 80–190 seconds on the machine that produced the solution, depending on load. On a laptop CPU allow 10–20 minutes, or use Colab with a GPU. MobileNetV3Small is several times faster and a few points less accurate.

## Rubric (20 points)

| area | points | what earns them |
|---|---|---|
| Data understanding | 2 | class counts per split; a baseline accuracy stated and used for comparison |
| Model | 3 | pretrained base frozen correctly; augmentation inside the model; trainable vs frozen weights reported |
| Training | 3 | validation set used for early stopping; learning curves plotted and interpreted honestly |
| Evaluation | 3 | test set used once; per-class precision/recall; confusion matrix read in words |
| Error analysis | 4 | at least 10 mistakes shown; grouped into causes; at least one cause that is the data's fault, not the model's |
| Confidence rule | 2 | threshold table; a recommendation with its trade-off stated |
| Saving | 1 | model saved, reloaded, predictions checked |
| Deployment note | 2 | plain language; names the weak spots; lists concrete checks before going live |

## Stretch tasks

- Allow up to 30 epochs. Does early stopping fire? Is the gain worth the time?
- Fine-tune the top 20 layers of the base with learning rate `1e-5` (08.9). Compare on the validation set before touching the test set.
- Train MobileNetV3Small on the same split and compare accuracy, model size on disk and prediction time per photo.
- Add a sixth class, "not a flower", using a few hundred photos of other ShopWave products. How does it change the confidence table?
- Rebuild the head and training loop in PyTorch with `torchvision.models.efficientnet_b0` (08.11).

## What the solution found

Read this only after your own attempt.

- Baseline (always "dandelion"): **24.5%**. EfficientNetB0 frozen + augmentation: **86.8%** test accuracy, training only 6,405 of its 4,055,976 weights.
- Validation accuracy levelled off around 0.90 while training accuracy kept rising; validation loss still edged down at epoch 15, so early stopping never fired.
- Dandelions are easiest (recall 0.948); roses (0.781) and tulips (0.817) are hardest, and they are mostly confused with each other (12 and 11 photos).
- Of 73 mistakes, 20 were made with over 80% confidence. They fall into groups: the flower is not the subject (a parade float, a breakfast table), mixed bouquets, unusual views (a dandelion from below, a closed bud, a seed head), a text overlay, and a few photos that look mislabelled.
- At a confidence threshold of 0.8 the model tags **80.4%** of photos on its own with **95.5%** accuracy; the other 20 in every 100 go to a person.
- The saved model is 17.1 MB and reloads with identical predictions.
- Not ready to go live yet: it must be tested on real supplier uploads, it needs a "not a flower" option, and low-confidence photos plus the team's corrections should be logged for retraining.

**Next:** [Module 09 · Language and generative AI](../../09-nlp-genai/).
