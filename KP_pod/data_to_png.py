import os

for i in range(512):
	os.system('python conv.py res\\{}.data png\\{}.png'.format(i,i))