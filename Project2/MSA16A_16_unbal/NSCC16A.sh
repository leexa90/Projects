  #!/bin/sh                                                                                                              
  #PBS -q normal                                                                                                         
  #PBS -l walltime=123:59:00                                                                                              
  #PBS -l nodes=1:ppn=2                                                                                                
  
#cd /home/users/astar/bii/leexa/machine_learning
cd $PBS_O_WORKDIR
module load tensorflow/1.0


python  new.py 0





