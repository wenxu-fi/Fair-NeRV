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


% 2D Fair-NeRV output

method="fnerv";
fnerv_opt=optimization_fnervF(X,classes,num_dims,method,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi,rng_seed);

%2D Fair-t-NeRV output

method="ftnerv";
ftnerv_opt=optimization_fnervF(X,classes,num_dims,method,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi,rng_seed);




