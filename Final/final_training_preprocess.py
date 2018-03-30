import jieba
import logging
import sys

def main():
    # jieba custom setting.
    jieba.set_dictionary('dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open(sys.argv[2], 'w', encoding='utf-8')
    with open(sys.argv[1], 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')
    output.close()

if __name__ == '__main__':
    main()