function [p,logp]=neighborprobone(X,sigma2,m);
   ndata=size(X,1);
   ndim=size(X,2);
   % compute squared distances
   Xdist=sum(X(m,:).^2,2) + sum(X.^2,2)' -2*X(m,:)*X'; 
   % ensure each point is not a neighbor of itself
   myinf=1e+16;
   Xdist(m)=Xdist(m)+myinf;
   % substract row minimums to preserve computational accuracy
   mindist=min(Xdist);
   logp=-(Xdist-mindist)/sigma2(m);
   p=exp(logp);
   % compute logp and p
   logp=logp-log(sum(p,2));
   p=p/sum(p,2);
end
