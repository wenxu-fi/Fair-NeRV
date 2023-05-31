function [P,logP]=neighborprob(X,sigma2)
  %This function compute neignbor probabilities
   ndata=size(X,1);
   % compute squared distances
  % Xdist=sum(X.^2,2); 
   %Xdist=repmat(Xdist,[1 ndata])+repmat(Xdist',[ndata 1])-2*(X)*(X'); %no need for repmat
   Xdist=squaredistance(X);
   % ensure each point is not a neighbor of itself
   myinf=1e+16;
   Xdist=Xdist+diag(myinf*ones(1,ndata));
   % substract row minimums to preserve computational accuracy
   mindist=min(Xdist,[],2);
   %size(Xdist)
   %size(mindist)
   logP=-(Xdist-repmat(mindist,[1 ndata]))./repmat(sigma2,[1 ndata]);
   P=exp(logP);
   % compute logP and P
   logP=logP-repmat(log(sum(P,2)),[1 ndata]);
   P=P./repmat(sum(P,2),[1 ndata]);
end
