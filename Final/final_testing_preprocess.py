import jieba
import sys
import pandas as pd
import numpy as np



def main():
	output_Q = open("test_Q.txt", 'w', encoding='utf-8')
	output_A = open("test_A.txt", 'w', encoding='utf-8')
	df_test = pd.read_csv("testing_data.csv")
	test = np.asarray(df_test)
	test_Q = test[:, 1]
	test_A = test[:, 2]
	test_Q = test_Q.tolist()
	test_A = test_A.tolist()
	new_test_Q=[]
	new_test_A=[]
	cnt=0
	for s in test_Q:
		#print(s)
		s = s.replace("A:", "")
		s = s.replace('B:', '')
		s = s.replace('C:', '')
		s = s.replace('A', '')
		s = s.replace(':', '')
		s = s.replace('\t', '')
		#new_test_Q.append(s)
		
		for i in range(6):
			cnt = cnt + 1
			output_Q.write(s+'\n')
	for s in test_A:
		#print(s)
		s = s.replace("A:", "")
		s = s.replace('B:', '')
		s = s.replace('C:', '')
		#print(s)
		new_test_A.append(s.split("\t"))
		#output_Q.write(s+'\n')
	for s in new_test_A:
		for line in s:
			output_A.write(line+'\n')
	output_Q.close()
	output_A.close()

if __name__ == '__main__':
    main()
