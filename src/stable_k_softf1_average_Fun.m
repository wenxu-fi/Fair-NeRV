function [stable_k,cluster_k_f1,stable_k_f1_average]=stable_k_softf1_average_Fun(K,nclasses,knnclass_f1,knncluster_f1)
    % This function is how we decide our fair scale threshold stable_k
    % f1(k) for high dimensionl  clusters and f1_avg  for all fair scales $k>stable_k$.  
    %{
    In extreme case for each point, its neighbors's probability belong to
    class i is p_i
    confusion matrix  
                   0       1
               0   N1*p1     N1*p2  
               1   N2*p1     N2*p2
    precision for the first class is N1*p1/(N1*p1+N2*p1)=N1*p1/N*p1=p1
    similarly recall is also p1 and so f1
    
    similarly f1 for second class is p2 
    Suitable for multiclass
    
    %}

    extreme_f1=1/nclasses; % sum(p_i)/nclasses
    
    %first k  such that class f1 close to random guessing extreme_f1
    smaller_extremef1_ind=find(knnclass_f1<=extreme_f1+0.01*(1-extreme_f1));
    greater_extremef1_ind=find(knnclass_f1>=extreme_f1-0.01*(1-extreme_f1));
    extremef1_neighbors_ind=intersect(smaller_extremef1_ind,greater_extremef1_ind);
    for k=1:K
        if all(ismember(k:K,extremef1_neighbors_ind))
           stable_k=k;
           break;
        end
    end  
   
    cluster_k_f1=knncluster_f1(stable_k); %f1(k)
    k_f1_sum=sum(knncluster_f1(stable_k:K));
    stable_k_f1_average=k_f1_sum/(K-stable_k+1); %f1_avg
end    