function squaredD=squaredistance(X)
    ndata=size(X,1);
    Xdist=sum(X.^2,2);
    Xdist=repmat(Xdist,[1 ndata])+repmat(Xdist',[ndata 1])-2*(X)*(X'); % this is square
    squaredD=Xdist-diag(Xdist);
end    
    
