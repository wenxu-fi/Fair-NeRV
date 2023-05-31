function soft_arr=softf1_uptoK_Fun(Y,ndata,nclasses,classes,minclasses,nclusters,clusters,minclusters,K)
          
    %compute distance matrix of Y
    dist_Y=pdist(Y);
    dis_matrix=squareform(dist_Y);
    
    %knearest_classes: each point's k nearest neighbor's class/cluster
    knearest_classes=zeros(ndata,K);
    knearest_clusters=zeros(ndata,K);
    for n=1:ndata
        dis_xn=dis_matrix(n,:);
        [B,I]=mink(dis_xn,K+1);
        ind=I(I~=n);
        knearest_classes(n,:)=classes(ind); 
        knearest_clusters(n,:)=clusters(ind); 
    end  
    % hard_arr: the first row is up to K hardknn f1 score for classes, the
    % second row is up to K hardknn f1 score for clusters. soft_arr is
    % similar result for soft knn. hard_weightedarr is macro weighted f1
    % score.
 
    soft_arr=zeros(2,K);
    

    for i=1:K
        soft_f1=fast_knnf1score_k(knearest_classes,ndata,nclasses,classes,minclasses,i);
        %fprintf("%f\n",hard_f1);
        %hard_arr(1,i)=hard_f1;
        soft_arr(1,i)=soft_f1;
        %hard_weightedarr(1,i)=hard_weightedf1;
        %soft_weightedarr(1,i)=soft_weightedf1;    
    end  

    for i=1:K
        soft_f1=fast_knnf1score_k(knearest_clusters,ndata,nclusters,clusters,minclusters,i);
        %fprintf("%f\n",hard_f1);
        %hard_arr(2,i)=hard_f1;
        soft_arr(2,i)=soft_f1;
       % hard_weightedarr(2,i)=hard_weightedf1;
       % soft_weightedarr(2,i)=soft_weightedf1;    
    end  
    
   
end