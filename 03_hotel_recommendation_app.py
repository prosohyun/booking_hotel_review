import webbrowser
import sys
import time
import re
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PyQt5.QtCore import QStringListModel
from gensim.models import Word2Vec
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from konlpy.tag import Okt


from_window = uic.loadUiType('./mainwidget.ui')[0]


class Exam(QWidget, from_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.df_review_one_sentence = pd.read_csv('./cleaned_review/onesentence_hotel_review_final.csv', index_col=0)
        self.stopwords = pd.read_csv('./stopwords_hotel.csv', index_col=0)
        self.Tfidf = TfidfVectorizer(sublinear_tf=True)
        self.Tfidf_matrix = self.Tfidf.fit_transform(self.df_review_one_sentence['review_one_sentence'])
        self.cmb_hotel_name.addItem('지역, 명소, 숙소명으로 찾아보세요  ⌕')
        titles = list(self.df_review_one_sentence['hotel_name'])
        for title in titles:
            self.cmb_hotel_name.addItem(title)
        self.cmb_hotel_name.currentIndexChanged.connect(self.cmb_hotel_name_slot)
        self.btn_exec.clicked.connect(self.btn_exec_clicked_slot)
        self.btn_area.clicked.connect(self.btn_area_clicked_slot)

        model = QStringListModel()
        model.setStringList(titles)
        completer = QCompleter()
        completer.setModel(model)
        self.le_name.setCompleter(completer)  #자동완성


    def getRecommendation(self, cosine_sim):
        simScores = list(enumerate(cosine_sim[-1]))
        print('1')
        simScores = sorted(simScores, key=lambda x: x[1], reverse=True)
        print('2')
        simScores = simScores[2:13] # 0번은 무조건 같은 영화기 때문에 1번부터
        print('3')
        hotelidx = [i[0] for i in simScores]

        RecHotellist = self.df_review_one_sentence.iloc[hotelidx]
        print('4')
        return RecHotellist.hotel_name

    def cmb_hotel_name_slot(self):
        title = self.cmb_hotel_name.currentText() #현재 출력되고 있는 문자열 읽어오기
        hotel_idx = self.df_review_one_sentence[self.df_review_one_sentence['hotel_name'] == title].index[0]

        cosine_sim = linear_kernel(self.Tfidf_matrix[hotel_idx], self.Tfidf_matrix)
        print(list(self.getRecommendation(cosine_sim)))
        self.lb_recommend.setText('\n'.join(list(self.getRecommendation(cosine_sim)[1:])))

    def btn_area_clicked_slot(self):
        webbrowser.open_new("https://www.yanolja.com/")


    def btn_exec_clicked_slot(self):
        word = self.le_name.text()
        embedding_model = Word2Vec.load('./model/word2VecModel_hotel.model') # Word2Vec 한 embedding_model을 활용해서 유사도 분석 후 영화 추천
        if word in embedding_model.wv.vocab:
            sim_words = embedding_model.wv.most_similar(word.split(' ')[0], topn=30)
            review = [word] * 11
            words = []
            for sim_word, _ in sim_words:  # sim_words는 튜플 형식으로 들어가있음 / 단어,simscore -> sim_word, _
                words.append(sim_word)
            for i in range(10):
                review = review + ([words[i]]*(10-i))
        else:
            review = [word]*11

        title_vec = self.Tfidf.transform([' '.join(review)])
        print('1')
        cosine_sim = linear_kernel(title_vec, self.Tfidf_matrix)
        print(len(cosine_sim[0]))
        print((self.getRecommendation(cosine_sim)))
        self.lb_recommend.setText('\n'.join(list(self.getRecommendation(cosine_sim))))



app = QApplication(sys.argv)
mainWindow = Exam()
mainWindow.show()
sys.exit(app.exec_())
