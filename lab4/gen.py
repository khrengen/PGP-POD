import random

n = 5000
print('{} {}\n'.format(n,n))
for i in range(n*n):
	print('{}'.format(random.uniform(-n,n)))