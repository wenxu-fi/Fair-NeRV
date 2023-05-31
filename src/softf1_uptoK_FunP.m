function soft_arr=softf1_uptoK_FunP(Y,ndata,nclasses,classes,minclasses,nclusters,clusters,minclusters,K)
    % This function give soft knn f1 score of classes(sensitive attribute) and clusters(high dimensional clusters)  for k =1,....,K     
    %compute distance matrix of Y
    %dist_Y=pdist(Y);
    %dis_matrix=squareform(dist_Y);
    dis_matrix=squaredistance(Y);
    
    %k nearest neighbors of sensitive attribute for k up to K
    knearest_classes=zeros(ndata,K);
    %k nearest neighbors of high dimensional clusters for k up to K
    knearest_clusters=zeros(ndata,K);
    
    for n=1:ndata
        dis_xn=dis_matrix(n,:);
        [B,I]=mink(dis_xn,K+1);
        ind=I(I~=n);
        knearest_classes(n,:)=classes(ind); 
        knearest_clusters(n,:)=clusters(ind); 
    end  
  
    % soft_arr: the first row is up to K soft knn f1 score for classes (sensitive attribute), the
    % second row is up to K soft knn f1 score for high dimensional clusters.
    soft_arr=zeros(2,K);
    for i=1:K
        soft_f1=fast_knnf1score_k(knearest_classes,ndata,nclasses,classes,minclasses,i);
        soft_arr(1,i)=soft_f1;  
    end  

    for i=1:K
        soft_f1=fast_knnf1score_k(knearest_clusters,ndata,nclusters,clusters,minclusters,i);
        soft_arr(2,i)=soft_f1; 
    end    
end