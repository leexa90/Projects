for i in `cat *csv`; do echo $i ;
molconvert --peptide $i mol -o ${i}.mol;done
