%This case show how apply our method to synthetica data
clear optimize_fnerv;
mex optimize_fnerv.cpp; %Fair-NeRV
clear optimize_ftnerv;
mex optimize_ftnerv.cpp; % Fair-t-NeRV
X= load('../dataset/syn/syn cluster/syn_test_x.csv');  % download data 
classes = load('../dataset/syn/syn cluster/syn_test_s.csv');  % download sensitive attribute, proprocess it so that  the minimum of the sensitive attribute value is 0.

learningrate=1;exaggeration=8;niters=1000;
tradeoff_intra=0.54;tradeoff_inter=0.91;
beta=0.04;gamma=0.69;coeffi=0.91;rng_seed=160;num_dims=2;method="fnerv";
%optimize
fnerv_opt=optimization_fnervF(X,classes,num_dims,method,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi,rng_seed);

method="ftnerv";
ftnerv_opt=optimization_fnervF(X,classes,num_dims,method,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi,rng_seed);

%plot with coloring of protected attribute and high dimensional clusters
test_output={fnerv_opt};

filename=fullfile('../dataset/syn/syn cluster','syn_test_cluster.csv');
clusters=load(filename);
clusters=clusters+1; %plot function settings' reason, clusters minimum value should be greater than 0. If your data's clusters' minimum value is greater than 0, no need do this
% plot output in two row, the first row is colored by sensitive classes and
% the second row is colored by high-dimensional clusters.
cluster_colors=Cluster_colorsF();
scatterplots_transparent_compact_labelL(test_output,classes,clusters,cluster_colors,2000,20);


% To check how we choose beat parameters  for
% different embeddings please check syn_summary.m 
