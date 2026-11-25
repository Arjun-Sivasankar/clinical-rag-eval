from clinical_rag.chunking import (
    Document,
    chunk_corpus,
    chunk_document,
    split_sentences,
    window_sentences,
)


def test_one_chunk_per_short_section():
    doc = Document("1", [("BACKGROUND", "Short background."), ("METHODS", "Short methods.")])
    chunks = chunk_document(doc, max_words=50)
    assert [c.section for c in chunks] == ["BACKGROUND", "METHODS"]
    assert [c.position for c in chunks] == [0, 1]
    assert all(c.pmid == "1" for c in chunks)


def test_long_section_split_on_sentences():
    text = " ".join(f"Sentence number {i} has five words." for i in range(20))  # 6 words each
    doc = Document("2", [("RESULTS", text)])
    chunks = chunk_document(doc, max_words=20)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 20 for c in chunks)
    assert " ".join(c.text for c in chunks) == text


def test_overlong_sentence_kept_whole():
    long = "word " * 50 + "end."
    assert window_sentences([long.strip()], 10) == [long.strip()]


def test_split_sentences_keeps_decimals():
    assert split_sentences("The OR was 1.5 in group A. Group B differed.") == [
        "The OR was 1.5 in group A.",
        "Group B differed.",
    ]


def test_corpus_ids_are_unique_and_sequential():
    docs = [Document(str(i), [("A", "x."), ("B", "y.")]) for i in range(3)]
    chunks = chunk_corpus(docs)
    assert [c.chunk_id for c in chunks] == list(range(6))


def test_empty_sections_skipped():
    assert chunk_document(Document("3", [("A", "  "), ("B", "text.")]))[0].section == "B"
