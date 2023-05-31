function sigma2 = findsigmas(X, nneighbors, entrtolerance)
   ndata = size(X,1);
   ndim = size(X,2);
   sigma2 = zeros(ndata,1);
   desiredentr = log(nneighbors);
   for m=1:ndata
      % starting entropy
      sigma2(m) = 1;
	  [p,logp] = neighborprobone(X,sigma2,m);
	  entr = entropy(p,logp);
	  
	  stepsize=1;
	  while (entr < desiredentr-entrtolerance)	  
	     % increase sigma until close enough
	     sigma2(m) = sigma2(m) + stepsize;
         [p,logp] = neighborprobone(X,sigma2,m);
         entrnew = entropy(p,logp);
		 %[m sigma2(m) entrnew desiredentr stepsize]
		 % check that we did not go too far
		 if (entrnew > desiredentr)
		    % restore old sigma and decrease stepsize
			sigma2(m) = sigma2(m) - stepsize;
			stepsize = stepsize*0.5;
		 else
		    % accept new sigma and increase stepsize
			entr = entrnew;
			stepsize = stepsize*1.1;
         end
      end
		 
	  stepsize=1;
	  while (entr > desiredentr+entrtolerance)
	     % decrease sigma until close enough
	     sigma2(m) = sigma2(m) - stepsize;
		 if (sigma2(m)>0)
            [p,logp] = neighborprobone(X,sigma2,m);
            entrnew = entropy(p,logp);
		 else
		     entrnew=-inf;
         end
			 
		 %[m sigma2(m) entrnew desiredentr stepsize]
		 % check that we did not go too far
		 if (entrnew < desiredentr)
		    % restore old sigma and decrease stepsize
			sigma2(m) = sigma2(m) + stepsize;
			stepsize = stepsize*0.5;
		 else
		    % accept new sigma and increase stepsize
			entr = entrnew;
			stepsize = stepsize*1.1;
         end
      end
	  
	  [m sigma2(m) exp(entr)];
   end
end
