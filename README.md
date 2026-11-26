# IR-TP2-Term-Weighting

**Course:** Information Retrieval (Master 2 – Computer Science)  
**Work Session 2:** Term Weighting (TF, Normalized TF, DF, IDF, and TF-IDF)

## Overview

This repository contains a from-scratch Python implementation of core term-weighting schemes in Information Retrieval, applied to a small pre-tokenized corpus with a fixed vocabulary:

`["information", "retrieval", "system", "model"]`

All weighting matrices and comparison tables are generated using custom Python functions (without relying on high-level library weighting functions for the main results) and subsequently verified against `scikit-learn`'s `TfidfVectorizer`.

---

## Repository Structure

```text
IR-TP2-Term-Weighting/
├── README.md
├── requirements.txt
├── tp2_term_weighting.ipynb
└── results/
    ├── tf matrix.csv
    ├── normalized tf matrix.csv
    ├── df_idf_comparison.csv
    ├── tfidf raw matrix.csv
    └── tfidf normalized matrix.csv
```

---

## Functions Implemented from Scratch

* **`compute_tf(document)`**  
  Counts raw term occurrences $tf(t, d)$ in a single tokenized document using `collections.Counter` and aligns counts to the fixed vocabulary.
* **`build_raw_tf_matrix(docs, vocab)`**  
  Constructs the document-term matrix of raw term frequencies as a `pandas.DataFrame`.
* **`compute_normalized_tf(document)`**  
  Computes length-normalized term frequency $tf_{\text{norm}}(t, d) = \frac{tf(t, d)}{|d|}$, dividing each raw count by the total number of tokens in the document.
* **`build_normalized_tf_matrix(docs, vocab)`**  
  Constructs the document-term matrix of normalized term frequencies.
* **`most_frequent_terms(raw_tf_df)`**  
  Identifies the term(s) with the highest raw frequency in each document, reporting all tied terms.
* **`compute_df(docs, vocab)`**  
  Computes document frequency $df(t)$ across the corpus, ensuring each term is counted at most once per document.
* **`compute_idf_1(df, n_docs)`**  
  Computes standard base-10 inverse document frequency:
  $$\text{IDF-1}(t) = \log_{10}\left(\frac{N}{df(t)}\right)$$
* **`compute_idf_2(df, n_docs)`**  
  Computes smoothed base-10 inverse document frequency:
  $$\text{IDF-2}(t) = \log_{10}\left(1 + \frac{N}{df(t)}\right)$$
* **`compute_tfidf_matrix(tf_matrix, idf_map)`**  
  Multiplies each term frequency entry by its corresponding IDF score ($w(t, d) = tf(t, d) \times idf(t)$) to produce both the raw TF-IDF and normalized TF-IDF matrices.
* **`highest_weighted_terms(tfidf_df)`**  
  Extracts the highest-weighted term(s) and their TF-IDF scores for every document.

---

## Main Conclusions

1. **IDF Balances Local Frequency with Global Rarity:**  
   In `D3`, `"retrieval"`, `"system"`, and `"model"` all have a raw TF of `1`, but `"model"` achieves the highest TF-IDF weight (`0.3010` vs. `0.1249`) because it appears in fewer documents ($df = 2$ vs. $df = 3$).
2. **Behavior When $df(t) = N$ (IDF-1 vs. IDF-2):**  
   Under **IDF-1**, a term appearing in every document yields $\log_{10}(N/N) = 0$, completely zeroing out its weight. Under **IDF-2**, the $+1$ inside the logarithm yields $\log_{10}(2) \approx 0.3010$, keeping a positive baseline weight and compressing differences between rare and common terms.
3. **Document-Length Normalization:**  
   Normalizing TF by document length prevents longer documents from receiving artificially inflated weights solely due to token count, making weights comparable across documents of varying lengths (e.g., `D3` with 3 tokens vs. `D1`, `D2`, `D4` with 4 tokens).
4. **Comparison with `scikit-learn`:**  
   While raw term frequencies match identically, `scikit-learn`'s `TfidfVectorizer` (even with `smooth_idf=False` and `norm=None`) produces larger numerical values than the course's IDF-1 because `scikit-learn` uses the **natural logarithm ($\ln$)** instead of **$\log_{10}$** and adds **`+ 1`** outside the logarithm: $\text{idf}_{\text{sklearn}}(t) = \ln(N / df(t)) + 1$.
