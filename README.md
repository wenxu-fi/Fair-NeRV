# Fair-NeRV
This repository contains codes for the experiments and algorithms Fair-NeRV/Fair-t-NeRV from the paper "Fair Neighbor Embedding" by Jaakko Peltonen, Wen Xu, Timo Nummenmaa and Jyrki Nummenmaa. The Fourtieth International Conference on Machine Learning (ICML 2023). Please cite this paper if you use the code.

All codes in this project are contributed and maintained by Wen Xu and Jaakko Peltonen. For questions, you may contact Wen Xu at wen.xu@tuni.fi/ wenxupine@gmail.com or Jaakko Peltonen at jaakko.peltonen@tuni.fi.

The implementations are in matlab with some parts in c++ for efficiency.

## Demonstration of usage in Matlab
Demonstration of usage in Matlab: check src/syn_case.m or run in folder src:

% download data and protected attribute

X= load('../dataset/syn/syn cluster/syn_test_x.csv');  

classes = load('../dataset/syn/syn cluster/syn_test_s.csv'); 

% set parameters

learningrate=1;exaggeration=8;niters=1000;
tradeoff_intra=0.54;tradeoff_inter=0.91;
beta=0.04;gamma=0.69;coeffi=0.91;rng_seed=160; num_dims=2; 
method="fnerv";

% 2D Fair-NeRV output
fnerv_opt=optimization_fnervF(X,classes,num_dims,method,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi,rng_seed);


For how we train 5 rounds of 3600 parameters settings of training data,  find fairnes scale threshold (stable_k) and corresponding f1, f1_avg of ,find the best parameters, with which embedding has maximum f1_avg, using thoses best parameters to run on test data set and get f1 and f1_avg for all methods, please see syn_summary.m (this file works on synthetic data). For other real datasets, implementations are similar, for example adult.summary, pima_summary. The file that includes commputation of fairness scale threshold k_fair ,f1_k, f1_avg is stable_k_softf1_average_Fun.m.


