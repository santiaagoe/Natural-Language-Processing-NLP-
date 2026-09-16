# Text Classification on 20 Newsgroups

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santiaagoe/Natural-Language-Processing-NLP-/blob/main/text-classification-project/Text_Classification.ipynb)

Multi-class text classification over the 20 Newsgroups corpus, comparing two text
representations under the same classifier. **The sparse baseline (TF-IDF) beat the dense
representation (word embeddings) by 13 accuracy points** — a result worth explaining, not
hiding.

---

## The task

Assign one of 20 mutually exclusive topics to a newsgroup post based on its text alone. The
model takes unstructured text as input and outputs a probability distribution over the 20
labels, predicting the most likely one.

The experiment is designed to isolate one variable: **the text representation**. The
classifier — logistic regression trained with SGD — is identical in both runs, so any
difference in performance is attributable to how the text is encoded.

---

## Data

[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/): 19,997 posts distributed almost
evenly across 20 topics (~1,000 each). The topics group into four broad themes: computing,
recreation, science, and society/religion.

The raw archive is hosted as a [release](https://github.com/santiaagoe/Natural-Language-Processing-NLP-/releases/tag/data-v1)
of this repository, so the notebook downloads it directly and needs no manual setup.

**Preprocessing:** each post is split at the first blank line to separate headers from body.
Only the body is used — the headers contain the newsgroup name, which would make the task
trivial.

| Split | Documents |
|---|---|
| Train (70%) | 13,997 |
| Test (30%) | 6,000 |

Stratified split, so every category keeps its proportion in both sets.

---

## Method

**Baseline — TF-IDF.** Sparse representation with 5,000 features and unigrams + bigrams. Each
document becomes a vector of term weights: frequent terms in the document, rare across the
corpus, get the highest values.

**Proposed improvement — dense embeddings.** Each document is represented as a 300-dimensional
vector: the IDF-weighted average of its spaCy word vectors (`en_core_web_md`), keeping only
alphabetic, non-stopword, in-vocabulary tokens.

**Classifier (identical in both).** Logistic regression trained with `SGDClassifier`
(`loss='log_loss'`), 50 epochs via `partial_fit`, constant learning rate of 0.01. Training
this way rather than with a closed solver allows plotting the cross-entropy loss curve on
train and test across epochs, which shows the convergence behaviour directly.

---

## Results

| Representation | Accuracy | Macro F1 |
|---|---|---|
| **Baseline — TF-IDF (5,000 features, 1–2 grams)** | **0.7547** | **0.7483** |
| Improvement — IDF-weighted spaCy embeddings (300 dims) | 0.6248 | 0.6135 |

**The proposed improvement made things worse.** The explanation is in what each
representation preserves.

Embeddings map semantically related words to nearby vectors — exactly the wrong property
here. Categories such as `alt.atheism`, `soc.religion.christian` and `talk.religion.misc`
share almost all of their semantic field and are separated by fine lexical differences.
Averaging word vectors over a document blurs precisely those differences. TF-IDF keeps each
discriminative term as its own dimension and weights it by how rare it is across the corpus,
which is what this task rewards.

There is a second factor: averaging destroys word order and document structure, and a
300-dimensional average is a far more aggressive compression than a 5,000-dimensional sparse
vector. Dense embeddings pay off when semantic generalization matters — short texts,
paraphrase, unseen vocabulary — not when the signal lives in specific terms.

---

## Limitations

- **The comparison is not fully controlled.** The embedding pipeline additionally removes
  stopwords, non-alphabetic tokens and out-of-vocabulary words. Part of the observed gap is
  attributable to preprocessing rather than to the representation itself.
- **Quoted text and signatures were not removed.** Headers are stripped, but replies still
  carry quoted fragments of earlier posts, and many messages end with signature blocks. This
  is a known source of leakage in 20 Newsgroups: quoted material drags vocabulary from its own
  category. Real accuracy on fully cleaned text is likely somewhat lower.
- **Bag-of-words loses context and word order.** TF-IDF ignores sentence structure and
  grammar entirely.
- **Linear decision boundaries.** Logistic regression assumes classes are linearly separable
  in feature space and cannot capture non-linear relationships between terms.
- **Static vocabulary.** The model only knows terms seen during training; new slang, typos or
  unseen words are ignored at inference time.
- **Single holdout split.** One 70/30 split with a fixed seed gives no estimate of variance.
  Cross-validation would make the 13-point gap more defensible.

---

## What I would try next

- Contextual embeddings (BERT or similar), which keep word order and disambiguate by context —
  the property that static averaged vectors throw away.
- Equalizing preprocessing across both pipelines to make the comparison fully controlled.
- Removing quotes and signatures to measure how much of the accuracy is leakage.

---

## Reproduce

**In Colab:** click the badge above. The notebook downloads both the dataset and the spaCy
model on its own — no manual setup.

**Locally:**

```bash
git clone https://github.com/santiaagoe/Natural-Language-Processing-NLP-.git
cd Natural-Language-Processing-NLP-/text-classification-project
pip install -r requirements.txt
python -m spacy download en_core_web_md
jupyter notebook Text_Classification.ipynb
```

> `requirements.txt` needs `spacy>=3.8` added — it is currently missing.

---

## Files

```
text-classification-project/
├── README.md
├── Text_Classification.ipynb
└── requirements.txt
```
