function [t,cluster_stable_f1average_arr]=f1average_parallel_ftnerv_local(data_title, sensstr,initial_Y,sigma2,P,nclasses,classes,nclusters,clusters,K,intra_start,intra_end,inter_end,beta_start,beta_end,gamma_start,gamma_end,contrast_coeffi_start,contrast_coeffi_end,niters,learningrate,exaggeration,n,rng_seed,output_folder)
    %This function computes  f1 and f1_avg scores of high dimensional
    %clusters for 3600 uniform sampling of parameters' spaces as  f1 score
    %of knn based prediction of sensitive attribute values grows near random guessing.
    
    t0=tic;
    %clear optimize_ftnerv;
    %mex optimize_ftnerv.cpp;
    ndata=size(initial_Y,1);
    minclasses=min(classes);
    minclusters=min(clusters);
    %uniform sampling of parameters spaces 
    intratemp=intra_start:(intra_end-intra_start)/(n-1):intra_end;
    betatemp=beta_start:(beta_end-beta_start)/(n-1):beta_end;
    gammatemp=gamma_start:(gamma_end-gamma_start)/(n-1):gamma_end;
    coeffitemp=contrast_coeffi_start:(contrast_coeffi_end-contrast_coeffi_start)/(n-1):contrast_coeffi_end;
    rng(rng_seed,'threefry');
    I1=randperm(n);
    I2=randperm(n);
    I3=randperm(n);
    I4=randperm(n);
    I5=randperm(n);
    
    interarray=zeros(n,1);
    for k=1:n
        intra=intratemp(I1(k));
        if intra<inter_end
            intertemp=intra:(inter_end-intra)/(n-1):inter_end;
            interarray(k)=intertemp(I2(k));
        else
            interarray(k)=intra;
        end    
    end  
   
    intraarray=intratemp(I1);
    betaarray=betatemp(I3);
    gammaarray=gammatemp(I4);
    coeffiarray=coeffitemp(I5);
    % initial array with columns tradeoff_intra, tradeoff_inter, beta,
    % gamma, contrast coefficient, stable_k, f1, f1_avg
    cluster_stable_f1average_arr=zeros(n,8);
    % parallel computation
    parfor k=1:n
        tradeoff_intra=intraarray(k);
        tradeoff_inter=interarray(k);
        beta=betaarray(k);
        gamma=gammaarray(k);
        coeffi=coeffiarray(k);

        %use current parameters to optimize
        optimized_Y=optimize_ftnerv(P,nclasses,classes,initial_Y,sigma2,tradeoff_intra,tradeoff_inter,beta,gamma,niters,learningrate,exaggeration,coeffi);

        filename=fullfile('../results',output_folder,'ftnerv','optresults',sprintf(data_title+sensstr+"ftnerv results tradeoff_intra(%.3f) tradeoff_inter(%.3f) beta(%.3f) gamma(%.3f) contrast_coeffi(%.3f).mat", tradeoff_intra,tradeoff_inter, beta,gamma,coeffi));
        parsave(filename,optimized_Y); %save optimized data
        
        %compute f1 scores of knn prediction of sensitive attribute values and high
        %dimensionl clusters for k=1,...,ndata-1      
        soft_arr=softf1_uptoK_FunP(optimized_Y,ndata,nclasses,classes,minclasses,nclusters,clusters,minclusters,K);
           
        %find the stable_k so that class f1score close to the random guessing, and the corresponding cluster f1score
        [first_k_soft,cluster_k_softf1,stable_k_softf1_average]=stable_k_softf1_average_Fun(K,nclasses,soft_arr(1,:),soft_arr(2,:));

         cluster_stable_f1average_arr(k,:)= [tradeoff_intra tradeoff_inter beta gamma coeffi first_k_soft cluster_k_softf1 stable_k_softf1_average];
    end
    %save to file
    filename=fullfile('../results',output_folder,'ftnerv','summary',sprintf(data_title+sensstr+"ftnerv cluster stable f1average outputs.mat"));
    save(filename,'cluster_stable_f1average_arr');
    t=toc(t0);
end