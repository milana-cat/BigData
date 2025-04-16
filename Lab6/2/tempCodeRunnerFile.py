# Инициализация Natasha
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(NewsEmbedding())
morph_vocab = MorphVocab()

# Предобработка
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = token.lemma
        if len(lemma) > 2 and lemma not in stop_words:
            lemmas.append(lemma)
    return " ".join(lemmas)
