"""Domain adaptation of the retrieval models on the corpus itself.

The models this system ships — bge-m3 for dense retrieval, the bge reranker —
were trained on general web text. Kazakh legal language is a long way from
that: fixed statutory terminology, numbered cross-references that carry the
operative meaning, and two languages that are legally equivalent but
lexically unrelated. The lever that actually moves retrieval quality here is
not a bigger model; it is the same model, fine-tuned on this corpus.

There is no labelled data, and no lawyer on hand to write it. The corpus is
its own supervision (``pairs.py``):

* an article's **title** is a short query its body answers;
* the sentence that **cites** article B is a natural-language query for B —
  with the citation itself cut out, so the model has to match meaning;
* the **Russian and Kazakh** texts of one article are two faces of one norm;
* a **defined term** is a query for the definition that follows it;
* the local LLM can write **questions** per article (``doc2query``), and the
  **query log** holds real questions with mechanically verified citations.

``finetune.py`` trains the embedder on those pairs with in-batch and mined
hard negatives; ``evaluate.py`` measures retrieval on a held-out set of
articles without a database, so base and fine-tuned models can be compared on
the same footing before either is put in front of a user.
"""
