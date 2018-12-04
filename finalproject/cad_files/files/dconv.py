from dolfin_utils import meshconvert
import sys
import os

ifilename = sys.argv[1]
ofilename = sys.argv[2]
iformat = None
print(ifilename, ofilename)
meshconvert.convert2xml(ifilename, ofilename, iformat=iformat)
os.system("dolfin-order %s" % ofilename)
