## Nomad2018 Predicting Transparent Conductors
[competition link](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors)

Competition aims to predict the formation energy (which is an indication of the stability of a new material) and the bandgap energy (which is an indication of the potential for transparency over the visible range).

### What i done
Feature engineering - with a total of 350 features. 

Crystal lattice properties (vol,angles,)
elemental properties
bond lengths
bond/dihedral angles
forces and energy (vdw and eswald)
Their mean,median,interquartile and discretized distribution are also used
Linear estimate using above parameters


Training - 
Subsampling of col ~ 20% and datapoints 40% was heavily done since it was found to improve performance. Intuitively the linear estimate gets the 'main trend' correct, and the other 350 features provide incremental improvements in the gradient boosting alogorithm. Since there were so many highly correlated features, subsampling at 20% gives features a "chance" of being selected. 

Predictions was made using 20 fold CV, with 1 validation and 1 test set. 

 CV error : 0.0487
 Public LB score : 0.0498
 Private LB score : ???
