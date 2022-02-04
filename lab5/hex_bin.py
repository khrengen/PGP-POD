#!/usr/bin/env python3

import binascii

string = str('0A000000'
 	'00000000' '00001041' '00000041'
 	'0000E040' '0000C040' '0000A040'
 	'00008040' '00004040' '00000040'
 	'0000803F')

with open('in.data', 'wb') as fl:
	fl.write(binascii.unhexlify(string.replace(' ', '')))