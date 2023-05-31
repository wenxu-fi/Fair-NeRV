function rankdistr = rankdistribution_perclass(P,nclasses,classes)

ndata=size(P,1);
minclass=min(classes);

classsubsets={};
for k=1:nclasses
    classsubsets{k} = find(classes==k-1+minclass);
end

rankdistr=zeros(nclasses,nclasses,ndata);
for i=1:ndata
  [y,I2] = sort(P(i,:),"descend");
  % I2 are the data indices in sorted order, but we need
  % the other way around: the order of each data item in the sort
  sortedpositions=[1:ndata];
  sortedpositions(I2)=[1:ndata];
  
  for k=1:nclasses
	rankdistr(classes(i)-minclass+1,k,sortedpositions(classsubsets{k}))=rankdistr(classes(i)-minclass+1,k,sortedpositions(classsubsets{k}))+1;
  end
  
%  for j=1:ndata,
%    rankdistr(classes(i)-minclass+1,classes(j)-minclass+1,I2(j))=rankdistr(classes(i)-minclass+1,classes(j)-minclass+1,I2(j))+1;
%  end;
end

end
