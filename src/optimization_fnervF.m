function optresult=optimization_fnervF(X,classes,num_dims,method,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi,rng_seed)
    %Optimize Fair-NeRV/Fair-t-NeRV
    
    %Input
    %X:data;
    %classes: sensitive attribute
    %num_dims: output dimension
    %tradeoff_intra, tradeoff_inter,beta,gamma,coeffiare parameters in cost
    %functions
    %niters: number of iterations (default 1000)
    %learning rate
    %exaggeration: exaggeration of input neighbors probability before
    %iteration 400
    %rng_seed: generator seed
    %method: "fnerv" of "ftnerv"
    
    
    if ~exist('niters', 'var') || isempty(niters)
        niters=1000;
    end
    if ~exist('learningrate', 'var') || isempty(learningrate)
        learningrate=1;
    end
    if ~exist('exaggeration', 'var') || isempty(exaggeration)
        exaggeration=8;
    end
    if ~exist('coeffi', 'var') || isempty(coeffi)
        coeffi=0.99;
    end
    if ~exist('rng_seed', 'var') || isempty(rng_seed)
        rng_seed=20;
    end
    
    nclasses=length(unique(classes));
    sigma2 = findsigmas(X, 20, 1e-4);
    P=neighborprob(X,sigma2);
    rng(rng_seed);ini_Y=randn(size(X,1),num_dims);
    if method=="fnerv"
        optresult= optimize_fnerv(P,nclasses,classes,ini_Y,sigma2,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi);
    elseif method=="ftnerv"
        optresult= optimize_ftnerv(P,nclasses,classes,ini_Y,sigma2,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi);
    end
end