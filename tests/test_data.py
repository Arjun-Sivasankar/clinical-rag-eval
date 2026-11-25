"""PubMedQA loading on a tiny fixture with the same parquet schema as `qiaojin/PubMedQA`."""

import pandas as pd

from clinical_rag.data import build_corpus, to_documents, to_queries


def _ctx(texts, labels):
    return {"contexts": texts, "labels": labels, "meshes": []}


LABELED = pd.DataFrame(
    {
        "pubid": [111, 222],
        "question": ["Do statins work?", "Is aspirin safe?"],
        "context": [
            _ctx(["Background text.", "Results text."], ["BACKGROUND", "RESULTS"]),
            _ctx(["Only one section."], ["OBJECTIVE"]),
        ],
        "long_answer": ["Yes, they do.", "Mostly."],
        "final_decision": ["yes", "maybe"],
    }
)
UNLABELED = pd.DataFrame(
    {
        "pubid": [222, 333],  # 222 duplicates a labelled abstract
        "question": ["dup", "Other?"],
        "context": [_ctx(["dup text."], ["X"]), _ctx(["Distractor."], ["METHODS"])],
        "long_answer": ["", ""],
    }
)


def test_to_documents_keeps_section_labels():
    docs = to_documents(LABELED, "pqa_labeled")
    assert docs[0].pmid == "111"
    assert docs[0].sections == [("BACKGROUND", "Background text."), ("RESULTS", "Results text.")]
    assert docs[0].text == "BACKGROUND: Background text.\nRESULTS: Results text."


def test_to_documents_label_mismatch_falls_back_to_unlabelled():
    df = LABELED.head(1).copy()
    df.at[0, "context"] = _ctx(["a.", "b."], ["ONLY_ONE"])
    assert [lab for lab, _ in to_documents(df, "x")[0].sections] == ["", ""]


def test_to_queries():
    q = to_queries(LABELED)
    assert list(q.columns) == ["qid", "question", "gold_pmid", "final_decision", "long_answer"]
    assert q["gold_pmid"].tolist() == ["111", "222"]


def test_build_corpus_dedupes_by_pmid_and_keeps_labelled_version(tmp_path):
    lab, unl = tmp_path / "lab.parquet", tmp_path / "unl.parquet"
    LABELED.to_parquet(lab)
    UNLABELED.to_parquet(unl)
    docs, queries = build_corpus({"pqa_labeled": lab, "pqa_unlabeled": unl})
    assert [d.pmid for d in docs] == ["111", "222", "333"]
    assert docs[1].split == "pqa_labeled"
    assert len(queries) == 2
