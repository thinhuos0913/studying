# from __future__ import division
# import nltk
# from nltk import FreqDist
# from nltk.corpus import brown

# news_text = brown.words(categories='news')
# fdist = nltk.FreqDist([w.lower() for w in news_text])
# print('FreqDist',fdist)
# from nltk.book import *
# nltk.download('gutenberg')
# Searching Text:
# print('Concordance:\n')
# text1.concordance("monstrous")
# print('Similar:\n')
# text1.similar("monstrous")
# text2.similar("monstrous")
# print('Common_context:\n')
# text2.common_contexts(["monstrous","very"])
# print("dispersion_plot:\n")
# # text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America", "liberty", "constitution"])
# print('Generate random text:\n')
# # text3.generate()
# # Counting Vocabularies:
# print(len(text3))
# # print(sorted(set(text3)))
# print(len(set(text3)))
# lexical_diversity: len(text3)/len(set(text3))
# print(len(text3) / len(set(text3)))
# sent1 = ['Call', 'me', 'Ishmael', '.']
# print(len(sent1))
# sent2 = ['The', 'family', 'of', 'Dashwood', 'had', 'long',
# 'been', 'settled', 'in', 'Sussex', '.']
# print(sent1 + sent2)
# print(sent1[2])
# print(sent2.index('Sussex'))
# print(sorted(sent2))
# token = set(sent2)
# print(token)
# print(sorted(token))
# fdist1 = FreqDist(sent1)
# print(fdist1)
# vocabulary1 = fdist1.keys()
# print(vocabulary1)

# MULTI-LABEL BINARIZER
# class sklearn.preprocessing.MultiLabelBinarizer(*, classes=None, sparse_output=False)
from lime import lime_text
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# label = {'0':'sci-fi', '1':'thriller','2':'comedy'}
# label = [{'sci-fi', 'thriller', 'comedy','horror'}]
label = [['amazing', 'sad', 'surprise']]

encoded = mlb.fit_transform(label)

# print(list(mlb.classes_))
print(encoded)

inv = mlb.inverse_transform(encoded)
print(inv)