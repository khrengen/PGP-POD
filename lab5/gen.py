import random
n = 10000
a = (random.uniform(-1000000, 1000000) for _ in range(n))
#a = [1 for _ in range(n)]
print(n)
for i in a:
	print(i)

#f = open('check.txt')
#a = -1000000
#for i in f:
#	if int(i) < a:
#		print('wa!!')
#	else:
#		a = int(i)
#f.close()