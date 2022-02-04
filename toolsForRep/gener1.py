import random
import sys

if len(sys.argv) < 2:
	print("введите n как входной арг")
	exit()

a = (random.uniform(-1000000000, 1000000000) for i in range(int(sys.argv[1])))

print(int(sys.argv[1]))
for i in a :
	print(i, end= ' ')