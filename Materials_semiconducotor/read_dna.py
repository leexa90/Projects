# open the file

import sys
name = sys.argv[1]

pdb_file = open(name,'r')
# read each line
new_file=  open('DNA_'+name,'w')
for line in pdb_file:
    if len(line ) >=19 and 'ATOM' in line[:4]:
        line = line[:18] + 'D' + line[19:]
        if line[19] == 'U':
            line = line[:19]+'T'+line[20:]
    if line[78] != 'H' and line[13:16].strip() != "O2'":
        new_file.write(line)
        print line,
##        if line[19] == 'T':
##            print line[:19]+' U'+line[20:],
##        else:
##            print line,

