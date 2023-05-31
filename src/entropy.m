function entr = entropy(P,logP)
   entr = -sum(P.*logP);
end
