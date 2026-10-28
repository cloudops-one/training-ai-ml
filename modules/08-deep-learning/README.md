# Module 08 · Deep Learning

**ShopWave asks:** *"We add 4,000 new products a month and someone tags every photo by hand: mug, cushion, lantern, bag. Can a model look at the photo and tag it?"*

**Time:** about 24 hours. **Prerequisites:** Module 02 (gradient descent), Module 03 (logistic regression, log loss), Module 04 (scaling, overfitting, validation).

## Where you are

Everything you have built so far starts from a **table**: one row per customer, one column per fact. Someone decided what the columns are. Recency, frequency, spend: a person chose those, and the model learned how to weigh them.

A product photo is not a table. It is 50,000 numbers, one per pixel per colour, and no single pixel means anything. Nobody can write down the columns that make a photo "a mug". You tried that kind of thing in lesson 00.1 with fraud rules, and it did not scale; with images it does not even start.

**Deep learning** solves this by learning the columns too. A neural network stacks simple units in layers; the early layers learn to detect edges and colours, the middle layers combine them into handles, rims and patterns, and the last layer decides "mug". Nobody writes those features. They are learned from labelled photos with the same gradient descent you met in lesson 02.4.

This module builds that idea from the ground up. You will compute one neuron by hand, run backpropagation on a two-layer network with real numbers, then use Keras to train networks on tables, on images and on sequences, and finally reuse a network trained by someone else on millions of photos.

One honest note before you start. **On ShopWave's churn table a neural network ties logistic regression** (test AUC 0.740 against 0.741, lesson 08.5), just as the boosted trees did in Module 05. Tables with a handful of meaningful columns rarely need deep learning. Images, text and audio do. That is where this module is heading.

## What you will be able to do afterwards

- Say when deep learning is the right tool and when a simpler model will do as well.
- Compute a neuron's output by hand, and show that one sigmoid neuron *is* logistic regression.
- Pick the right activation function and loss for a task, and compute each by hand.
- Run backpropagation by hand on a small network: forward pass, chain rule, one update, lower loss.
- Build, train, evaluate and save networks with Keras, and count their parameters by hand.
- Read training curves and fix a network that will not learn or that overfits: learning rate, dropout, L2, early stopping.
- Explain convolution by computing one on a 5×5 image, and train a CNN on clothing photos.
- Use data augmentation and transfer learning to get good results from a few hundred photos.
- Explain how recurrent networks handle sequences, and why Transformers replaced them.
- Read the same model written in PyTorch *(optional)*.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 08.1 | [When deep learning wins, and when it does not](01-when-deep-learning-wins/) | 0.5 h |
| 08.2 | [One neuron, by hand and in NumPy](02-one-neuron/) | 2 h |
| 08.3 | [Activation functions and loss functions](03-activations-and-losses/) | 1.5 h |
| 08.4 | [Backpropagation on a two-layer network, by hand](04-backpropagation-by-hand/) | 3 h |
| 08.5 | [Your first network in Keras](05-first-keras-network/) | 2 h |
| 08.6 | [Training well: epochs, batches, learning rate, dropout, early stopping](06-training-well/) | 2 h |
| 08.7 | [Images as numbers; convolution by hand on a 5×5 image](07-images-and-convolution/) | 2 h |
| 08.8 | [CNNs in Keras on Fashion-MNIST](08-cnns-fashion-mnist/) | 2 h |
| 08.9 | [Data augmentation and transfer learning](09-augmentation-transfer-learning/) | 2.5 h |
| 08.10 | [Sequences: RNNs, LSTMs and why they led to Transformers](10-sequences-rnn-lstm/) | 1.5 h |
| 08.11 | [The same model in PyTorch](11-pytorch/) *(optional)* | 1.5 h |
| 08.P | [Mini-project: ShopWave product image classifier](project/) | 3.5 h |

## Your computer is enough

Every notebook in this module runs on a laptop CPU, including Windows machines where TensorFlow cannot use the graphics card, and fits comfortably in 8 GB of memory. Training runs take seconds to a few minutes. A GPU (or Colab's free one) makes lessons 08.8–08.9 faster but is never required.

Install the libraries once, before lesson 08.2:

```bash
pip install -r setup/requirements-deep-learning.txt
```

Neural networks start from random weights, so your numbers will differ slightly from the ones printed here, by a few thousandths, depending on your machine and library versions. Every notebook sets a random seed so that re-running on the *same* machine gives the same answer.

## The data in this module

- **By hand:** tiny examples typed into the notebooks: one customer, a 2-input network, a 5×5 image.
- **Tables:** the ShopWave churn table from Module 03 (3,317 customers, 43.6% churned, test AUC 0.741 to beat).
- **Images:** scikit-learn's 8×8 handwritten digits (08.1), Fashion-MNIST, 70,000 small clothing photos that come with Keras (08.7–08.9), and a small product-photo set for the project.

## The maths in this module

Four ideas, each computed by hand once:

- **A neuron**: a weighted sum plus a bias, squashed by an activation: $a = f(w \cdot x + b)$.
- **A layer is a matrix multiplication**: three customers × three features times a 3×4 weight matrix gives 3×4 outputs.
- **The chain rule**: how much the loss changes when one weight changes is a product of small, local slopes, passed backwards through the network. That is all backpropagation is.
- **Convolution**: slide a small grid of weights over an image, multiply and add at every position.
