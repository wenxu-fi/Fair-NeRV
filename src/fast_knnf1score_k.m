function soft_f1=fast_knnf1score_k(knearest_sens,ndata,nclasses,classes,minclasses,k)
    % This function computes f1 of soft knn classification of classes for k
    
    %initial soft confusion matrix, row corresponds to true classes and
    %columns corresponds to predicted classes
    soft_matrix=zeros(nclasses,nclasses);
    for n=1:ndata
        n_sens=classes(n);        
        sens_I=knearest_sens(n,1:k);
        weight=zeros(1,nclasses);
        for m=1:nclasses
            weight(1,m)=sum(sens_I==m-1+minclasses)/k;
            soft_matrix(n_sens-minclasses+1,m)=soft_matrix(n_sens-minclasses+1,m)+weight(1,m);
        end  
    end    
       
    soft_precision=zeros(nclasses,1);
    soft_recall=zeros(nclasses,1);
    for m=1:nclasses       
       if sum(soft_matrix(:,m))==0         
          soft_precision(m,1)=0.5;           
       else
           soft_precision(m,1)=soft_matrix(m,m)/sum(soft_matrix(:,m));
       end 
       
       if sum(soft_matrix(m,:))==0    
           soft_recall(m,1)=0.5;     
       else
           soft_recall(m,1)=soft_matrix(m,m)/sum(soft_matrix(m,:));
       end    
    end  
        
    soft_macro=2*soft_precision.*soft_recall./(soft_precision+soft_recall);
    soft_macro(isnan(soft_macro))=0;
    soft_f1=sum(soft_macro)/nclasses; %soft f1
    
end
    