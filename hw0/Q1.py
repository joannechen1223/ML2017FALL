'''
ML_HW0_Q1
Date:2017.09.16
Author:b04701232 joannechen1223
'''
#!/usr/bin/env python3
import sys
f1 = open(sys.argv[1], 'r')
original = f1.read()
f1.close()
original = original.strip('\n')
after_sep = original.split(' ')
length = len(after_sep)
arr_str = [after_sep[0]]
arr_num = [1]
for i in range(1, length):
	check = 0
	for j in range(len(arr_str)):
		if after_sep[i]==arr_str[j]:
			arr_num[j]+=1
			check = 1
			break
	if check==0:	
		arr_str.append(after_sep[i])
		arr_num.append(1)
f2 = open('Q1.txt', 'w')
l = len(arr_str)
for i in range(l):
	f2.write("{} {} {}".format(arr_str[i],i,arr_num[i]))
	if(i<l-1):
		f2.write('\n')
f2.close()