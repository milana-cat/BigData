try:
    similar_words = model.wv.most_similar("пустота", topn=10)
    print("Ближайшие слова к 'пустота':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.3f}")
except KeyError:
    print("Слово 'пустота' не найдено в словаре модели. Попробуй другое.")
