GloVe 6B, 100 вимірів (Pennington, Socher, Manning 2014; Wikipedia 2014 + Gigaword 5),
ліцензія Public Domain Dedication and License (PDDL).
Оригінал — 400 000 слів, 128 МБ: gensim.downloader.load('glove-wiki-gigaword-100').
Тут 24972 слів: 20000 найчастіших + усі слова наших корпусів і тестів, які є в GloVe.
Вектори ті самі, обрізано лише словник. Формат word2vec binary:
    KeyedVectors.load_word2vec_format('assets/glove/glove_100d.bin', binary=True)
