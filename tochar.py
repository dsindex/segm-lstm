#!/bin/env python
#-*- coding: utf8 -*-

import sys
import os
from   optparse import OptionParser

# --verbose
VERBOSE = 0

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
	(options, args) = parser.parse_args()
	if options.verbose == 1 : VERBOSE = 1
	
	i = 0
	while 1 :
		try : line = sys.stdin.readline()
		except KeyboardInterrupt : break
		if not line : break
		line = line.strip()
		if not line : continue
		line = line.decode('utf-8')
		print ' '.join(line).encode('utf-8')

		i += 1
