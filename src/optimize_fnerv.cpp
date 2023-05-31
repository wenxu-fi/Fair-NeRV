// This is the implementation of the Fair-NeRV algorithm.
//In our algorithm, we have data with categorical sensitive attribute.  Before using this you should preprocess data so that minimum sensitive attribute value is 0
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mex.h"
#define max(x,y) ((x)>(y)?(x):(y))
#define MINCLASS 0

#define MATRIXELEMENT(ROW,COL,NROWS,NCOLS) (COL*NROWS+ROW)

static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// Compute neighbor probability 
void compute_neighborprob
(
    int ndata, 
    int ndim,   
    int nclasses,
	double *classes,
    double *X, 
    double *Xdist,
    double *distRowMin,
    double *distClassRowMin,
    double *logExpSum,
    double *ExpClassSum,
    double *sigma2, 
    double *P, 
    double *logP
)
{
    int i, j, k;
	int cellindex;
	
	// Large value (not quite infinity to avoid some computation errors)
    double myinf=1e+16;

    for (int i=ndata*nclasses-1;i>=0;i--)
    {
		distClassRowMin[i]=myinf;
	}


    // compute squared distances
    for (i = ndata-1; i >= 0; i--)
    {
		// Find the row minimum so far
		double rowminimum = myinf;
		for (j = ndata-1; j > i; j--)
		{
			int sens_j=(int)classes[j]-MINCLASS;			
			double tempval = Xdist[j*ndata+i];
			if (tempval < rowminimum)
			{
				rowminimum = tempval;
			}
			if (tempval<distClassRowMin[sens_j*ndata+i])
			{
				distClassRowMin[sens_j*ndata+i]=tempval;
			}
			
		}
		
		// Compute the rest of the distances in the row
 	    for (j = i-1; j >= 0; j--) 
	    {
			int sens_j=(int)classes[j]-MINCLASS;			
		    double tempdist = 0;
		    for (k = (ndim-1)*ndata; k >= 0; k-=ndata)
		    {
			    double tempdiff = X[k+i]-X[k+j];
			    tempdist += tempdiff*tempdiff;
				
		    }
			// Fill in distance value and its symmetric pair
		    Xdist[j*ndata+i] = tempdist;
		    Xdist[i*ndata+j] = tempdist;
			
			
			// Update row minimum
			if (tempdist < rowminimum)
			{
				rowminimum = tempdist;
			}
			if (tempdist<distClassRowMin[sens_j*ndata+i])
			{
				distClassRowMin[sens_j*ndata+i]=tempdist;
			}
	    }
	    
	    distRowMin[i]=rowminimum;
		
			
		// Substract row minimum to preserve computational accuracy
		/*
 	    for (j = ndata-1; j >= 0; j--) 
	    {
		    Xdist[j*ndata+i] -= rowminimum;
		}
		*/
		
        
		
		// Ensure each point is not a neighbor of itself
		Xdist[i*ndata+i] = myinf;

	 
    }	
    
    for (int i=ndata*nclasses-1;i>=0;i--)
    {
		ExpClassSum[i]=0;
	}
    for (i=ndata-1;i>=0;i--)
    {
		double tempsigma2=sigma2[i];
		double exp_sum = 0;
		double *Xdist_rowi = Xdist+i;
		double *logP_rowi = logP+i;
		double *P_rowi = P+i;
 	    for (j = ndata-1; j >= 0; j--) 
	    {
		  int s_j=(int)classes[j]-MINCLASS;
		  ExpClassSum[s_j*ndata+i]+=exp(-(Xdist_rowi[j*ndata]-distClassRowMin[s_j*ndata+i])/tempsigma2);
		  
		  double exp_temp = -(Xdist_rowi[j*ndata]-distRowMin[i])/tempsigma2;
		  
		  logP_rowi[j*ndata] = exp_temp;
		  exp_temp = exp(exp_temp);
		  P_rowi[j*ndata] = exp_temp;
		  exp_sum += exp_temp;
		 }
        
		logExpSum[i] = log(exp_sum);
 	    for (j = ndata-1; j >= 0; j--) 
	    {
			logP_rowi[j*ndata] -= logExpSum[i];
			P_rowi[j*ndata] /= exp_sum;			
		}
		
        // Ensure point is not a neighbor of itself, but leave 
		// nonzero logprob to avoid nans in some places		
		P_rowi[i*ndata] = 0;
		logP_rowi[i*ndata] = -DBL_MAX;
    }
    
	

}




// computation of cost function
void compute_totalcost
(
    int ndata,
	int ndim,
    double *P,
	double *Q,
	double *logP,
	double *logQ,
	int nclasses,
	double *classes,
	double *desiredclassprobs, // Preallocated temp array for computation
	double *classprobs,        // Preallocated temp array for computation
	double tradeoff_intra,
	double tradeoff_inter,
	double beta,             // Fairness vs NeRV cost balance parameter
	double gamma,            //balance parameter of recall vs precision of desiredclassprobs and low dimensional classprob
	double *output_cost,
	double *output_reccost_intra,
	double *output_preccost_intra,
	double *output_reccost_inter,
	double *output_preccost_inter,
	double *output_faircost1,
	double *output_faircost2
)
{
    int i, j, k;   
	double cost, reccost_intra, preccost_intra, reccost_inter, preccost_inter;
	double faircost, faircost1, faircost2;
   
    // Compute Nerv cost: recall and precision KL divergences
	reccost_intra = preccost_intra = reccost_inter = preccost_inter = 0;
	
	for (i = ndata - 1; i >= 0; i--)
	{
		for (j = ndata - 1; j >= 0; j--)
		{
			int tempofs_jndatai = j * ndata + i;
			if (classes[i] == classes[j])
			{
				reccost_intra += P[tempofs_jndatai] * (logP[tempofs_jndatai] - logQ[tempofs_jndatai]) + Q[tempofs_jndatai] - P[tempofs_jndatai];
				preccost_intra += Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai]) + P[tempofs_jndatai] - Q[tempofs_jndatai];
			}
			else
			{
				reccost_inter += P[tempofs_jndatai] * (logP[tempofs_jndatai] - logQ[tempofs_jndatai]) + Q[tempofs_jndatai] - P[tempofs_jndatai];
				preccost_inter += Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai]) + P[tempofs_jndatai] - Q[tempofs_jndatai];
			}
			if ((isnan(reccost_intra)) || (isnan(preccost_intra)) || (isnan(reccost_inter)) || (isnan(preccost_inter)))
			{
				mexPrintf("i: %d j: %d, P %e, Q %e, logP %e, logQ %e\n", i, j, P[tempofs_jndatai], Q[tempofs_jndatai], logP[tempofs_jndatai], logQ[tempofs_jndatai]);
				//*output_cost = 0.0/0.0;
				break;
			}
		}
	}
	
	
	// Compute fairness cost, faircost1: KL(desiredclassprobs,classprobs):KL(,r_i). faircost2:KL(classprobs,desiredclassprobs)
    faircost1 = 0;
	faircost2 = 0;
    
	
    for (i = ndata-1; i >= 0; i--)
    {		
		double *classprobs_i = classprobs+i;
		for (j = nclasses-1; j >= 0; j--)
		{
			classprobs_i[j*ndata] = 0;
		}
		// Compute class distribution around point i
		for (j = ndata-1; j >= 0; j--)
		{
			classprobs_i[((int)classes[j]-MINCLASS)*ndata] += Q[j*ndata+i];
		}
        
		// Compute KL to overall class distribution
		double klvalue1 = 0;
		double klvalue2 = 0;
		for (j = nclasses-1; j >= 0; j--)
		{          			
			klvalue1 += desiredclassprobs[j * ndata + i] * (log(max(desiredclassprobs[j * ndata + i], DBL_MIN)) - log(max(classprobs_i[j * ndata],DBL_MIN)));
			klvalue2 += classprobs_i[j * ndata] * (log(max(classprobs_i[j * ndata],DBL_MIN)) - log(max(desiredclassprobs[j * ndata + i], DBL_MIN)));
			if ((isnan(klvalue1)) | (isnan(klvalue2)))
			{
				mexPrintf("i %d, j %d, desired %f, class %f\n", i, j, desiredclassprobs[j * ndata + i], classprobs_i[j * ndata]);
				break;
			}
		}
		//faircost1 += klvalue;
		faircost1 += klvalue1; // 
	    faircost2 += klvalue2;
	    
	}
	
	faircost = gamma * faircost1 + (1 - gamma) * faircost2;
	
   
	
	// Compute total cost
	cost = beta * (tradeoff_intra * reccost_intra + (1 - tradeoff_intra) * preccost_intra
		+ tradeoff_inter * reccost_inter + (1 - tradeoff_inter) * preccost_inter) + (1 - beta) * faircost;
    

    // Assign outputs
    *output_cost = cost;
	*output_reccost_intra = reccost_intra;
	*output_preccost_intra = preccost_intra;
	*output_reccost_inter = reccost_inter;
	*output_preccost_inter = preccost_inter;
	*output_faircost1 = faircost1;
	*output_faircost2 = faircost2;
}

/*
double grad_denom(int ndata,int i,int j,double *Ydist,double *classes,double *sigma2)
{
	double grad_denom_sum=0;
	double *dist_i=Ydist+i;	
	double sens_j=(int)classes[j]-MINCLASS;
	for (int k=ndata-1;k>=0;k--)
	{
		double sens_k=(int)classes[k]-MINCLASS;
		if(k!=i && sens_k==sens_j)
		{grad_denom_sum+=exp(-(dist_i[k*ndata]-dist_i[j*ndata])/sigma2[i]);
		}
	}
	return grad_denom_sum;
	}
*/	
// gradient computation of cost function 	
void compute_grads
(
    int ndata,
	int ndim,
	double *Y,
	double *Ydist,
	double *distRowMin,
	double *distClassRowMin,
	double *logExpSum,
	double *ExpClassSum,
    double *P,
	double *Q,
	double *logP,
	double *logQ,
	int nclasses,
	double *classes,
	double *desiredclassprobs, 
	double *classprobs,        
	double *temp_sumterms,     
	double *sigma2,
	double tradeoff_intra,
	double tradeoff_inter,
	double beta,
	double gamma,
	double *nerv_intra,
	double *nerv_inter,
	double *grads
)
{
	int i, j, k;

	// Initialize gradient to zero
	for (i=ndata*ndim-1; i >= 0; i--)
	{
		grads[i] = 0;
	}
	
	//Initialize and compute  nerv_intra and nerv_inter
	// nerv_intra is {sum_{j\in S_i^{\in}}(tradeoff_intra*(Q_ij-P_ij)+(1-tradeoff_intra)Q_ij*(logQ_ij-logP_ij))}_{i=1}^{ndata}
	// nerv_inter is {sum_{j\in S_i^{\notin}}(tradeoff_inter*(Q_ij-P_ij)+(1-tradeoff_inter)Q_ij*(logQ_ij-logP_ij))}_{i=1}^{ndata}

	for (i = ndata - 1; i >= 0; i--)
	{
		nerv_intra[i] = nerv_inter[i] = 0;
	}
	for (i = ndata - 1; i >= 0; i--)
	{
		for (j = ndata - 1; j >= 0; j--)
		{
			int tempofs_jndatai = j * ndata + i;
			if (classes[j] == classes[i])
			{
				nerv_intra[i] += tradeoff_intra * (Q[tempofs_jndatai] - P[tempofs_jndatai]) +
					(1 - tradeoff_intra) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai]);
			}
			else
			{
				nerv_inter[i] += tradeoff_inter * (Q[tempofs_jndatai] - P[tempofs_jndatai]) +
					(1 - tradeoff_inter) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai]);
			}
		}
	}
	
	// Compute class distributions for fairness gradient, denoted as r_i in the paper
	for (i=ndata-1; i >= 0; i--)
	{
		// Compute class distribution around point i
		double *classprobs_i = classprobs+i;
		for (j = nclasses-1; j >= 0; j--)
		{
			classprobs_i[j*ndata] = 0;
		}
		for (j = ndata-1; j >= 0; j--)
		{
			classprobs_i[((int)classes[j]-MINCLASS)*ndata] += Q[j*ndata+i];
		}
	}


	//compute sequence of D_kl(r_i,\rho_i)
	for (i = ndata - 1; i >= 0; i--)
	{
		temp_sumterms[i] = 0;
		double* classprobs_i = classprobs + i;
		double* desiredclassprobs_i = desiredclassprobs + i;
		for (j = nclasses - 1; j >= 0; j--)
		{
			temp_sumterms[i] += classprobs_i[j * ndata] * ((distRowMin[i]-distClassRowMin[i+j*ndata])/sigma2[i]+log(ExpClassSum[i+j*ndata])-logExpSum[i]- log(max(desiredclassprobs_i[j * ndata], DBL_MIN))); 
			
			//temp_sumterms[i] += classprobs_i[j * ndata] * (log(max(classprobs_i[j * ndata],DBL_MIN)) - log(max(desiredclassprobs_i[j * ndata], DBL_MIN)));
			//mexPrintf("i %d %f\n", i, j, temp_sumterms[i]);
		}

	}
	//gradient
	for (i=ndata-1; i >= 0; i--)
	{
		double sigma2_i = sigma2[i];
		double* classprobs_i = classprobs + i;
		double* desiredclassprobs_i = desiredclassprobs + i;
		int s_i = (int)classes[i] - MINCLASS;
		for (j=ndata-1; j >= 0; j--)
		{
			int s_j = (int)classes[j] - MINCLASS;
			int tempofs_jndatai = j*ndata+i;
			int tempofs_indataj = i*ndata+j;
			
			// gradient of classnerv
			double temp_multiplier1 =  nerv_intra[i] * Q[tempofs_jndatai]/sigma2_i
				+ nerv_intra[j] * Q[tempofs_indataj]/sigma2[j]
			     +nerv_inter[i] * Q[tempofs_jndatai] / sigma2_i
				+ nerv_inter[j] * Q[tempofs_indataj] / sigma2[j];	
			
			
			if (classes[j] == classes[i])
			{
				temp_multiplier1 -=  (tradeoff_intra * (Q[tempofs_jndatai] - P[tempofs_jndatai])
					+ (1 - tradeoff_intra) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai]))/sigma2_i
					+  (tradeoff_intra * (Q[tempofs_indataj] - P[tempofs_indataj])
						+ (1 - tradeoff_intra) * Q[tempofs_indataj] * (logQ[tempofs_indataj] - logP[tempofs_indataj]))/sigma2[j];
						
					
			}
			else{
				temp_multiplier1 -=  (tradeoff_inter * (Q[tempofs_jndatai] - P[tempofs_jndatai])
					+ (1 - tradeoff_inter) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai])) / sigma2_i
					+  (tradeoff_inter * (Q[tempofs_indataj] - P[tempofs_indataj])
						+ (1 - tradeoff_inter) * Q[tempofs_indataj] * (logQ[tempofs_indataj] - logP[tempofs_indataj])) / sigma2[j];	
				}
				
		    //gradient for fair part		
		    double temp_multiplier2=(gamma*(Q[tempofs_jndatai]-desiredclassprobs_i[s_j*ndata] * exp(-(Ydist[tempofs_jndatai]-distClassRowMin[s_j*ndata+i])/sigma2_i)/ExpClassSum[i+s_j*ndata])
		    +(1-gamma)*((distRowMin[i]-distClassRowMin[i+s_j*ndata])/sigma2_i+log(ExpClassSum[i+s_j*ndata])-logExpSum[i]- log(max(desiredclassprobs_i[s_j * ndata], DBL_MIN)) - temp_sumterms[i]) * Q[tempofs_jndatai])/sigma2_i
		    +(gamma*(Q[tempofs_indataj] - desiredclassprobs[s_i * ndata + j]*exp(-(Ydist[tempofs_indataj]-distClassRowMin[s_i*ndata+j])/sigma2[j]) /ExpClassSum[j+s_i*ndata] )
		    +(1-gamma)*((distRowMin[j]-distClassRowMin[j+s_i*ndata])/sigma2[j]+log(ExpClassSum[j+s_i*ndata])-logExpSum[j]- log(max(desiredclassprobs[s_i * ndata + j], DBL_MIN)) - temp_sumterms[j]) * Q[tempofs_indataj])/sigma2[j];
			double *Y_rowj = Y+j;
			double *Y_rowi = Y+i;
			double *grads_rowi = grads+i;
		    for (k=(ndim-1)*ndata; k >= 0; k-=ndata)
		    {
			    grads_rowi[k] +=(beta*temp_multiplier1-(1-beta)*temp_multiplier2)*(Y_rowi[k]-Y_rowj[k]);
		    }
		}
	}

	for (i=ndata*ndim-1; i >= 0; i--)
	{
		grads[i] *=2 ;
	}
	
	
	
}


void zeroMean
(
	double* Y,
	double* Y_mean,
	int ndata,
	int ndim
)
{
	for (int j = ndim - 1; j >= 0; j--)
	{
		Y_mean[j] = 0;
	}
	for (int i = ndata-1; i >=0; i--)
	{
		for (int j = ndim - 1; j >= 0; j--)
		{
			Y_mean[j] += Y[j * ndata + i];
		}
	}
	for (int j = ndim - 1; j >= 0; j--)
	{
		Y_mean[j] /= ndata;
	}

	for (int i = ndata - 1; i >= 0; i--)
	{
		for (int j = ndim - 1; j >= 0; j--)
		{
			Y[j*ndata+i]-= Y_mean[j];
		}
	}

}

void optimize_embedding
(
    int ndata,
	int ndim,
    double *input_P, 
	int nclasses,
	double *classes,
	double *Yinitial, 
	double *Y,
	double *sigma2, 
	double tradeoff_intra,
	double tradeoff_inter,
	double beta,
	double gamma,
	double contrast_coeffi,
	int niters, 
	double learning_rate, 
	double exaggeration
)
{
	int iter, i, j, k;
	double cost, reccost_intra, preccost_intra, reccost_inter, preccost_inter, faircost1, faircost2;
	double momentum = .5, final_momentum = .8;
	int momentumChange = 250;
	int exaggerationStop = 100;
	// Start from initial position
	for (i=ndata-1; i >= 0; i--)
	{
		for (j=ndim-1; j >= 0; j--)
		{
			Y[j*ndata+i] = Yinitial[j*ndata+i];
		}
	}
    

    // Compute  overall class probabilities (u in paper)
    double *overallclassprobs = (double *)mxMalloc(nclasses*sizeof(double));
	for (j = nclasses-1; j >= 0; j--)
	{
	    overallclassprobs[j] = 0;
	}
    for (i = ndata-1; i >= 0; i--)
    {
	    overallclassprobs[(int)classes[i]-MINCLASS] += 1;
	}
	for (j = nclasses-1; j >= 0; j--)
	{
	    overallclassprobs[j] /= ndata;
	}	

    // desiredclassprobs (\rho_i in paper)
	double *desiredclassprobs = (double*)mxMalloc(ndata * nclasses * sizeof(double));
	for (i = 0; i < ndata; i++)
	{
		int sens_i = (int)classes[i] - MINCLASS;
		//mexPrintf("i %d sens_i %d\n", i, sens_i);
		for (j = 0; j < nclasses; j++)
		{
			if (sens_i == j)
			{
				desiredclassprobs[j * ndata + i] = 1-contrast_coeffi;

			}
			else { desiredclassprobs[j * ndata + i] = contrast_coeffi*overallclassprobs[j]/(1-overallclassprobs[sens_i]); }
			//mexPrintf("i%d j %d %f ",i, j, desiredclassprobs[j * ndata + i]);
		}
		//mexPrintf("\n");
	}
    
	
    // Create a preallocated array for class probabilities (r_i in paper)
	
    double *classprobs = (double *)mxMalloc(ndata*nclasses*sizeof(double));
	double *Y_mean= (double*)mxMalloc(ndim*sizeof(double));

    // Create a preallocated array for sum terms needed in fairness gradient
    double *temp_sumterms = (double *)mxMalloc(ndata*sizeof(double)); // on 1 no need this

	
	// create two prelocated arrays for gradient of nerv cost
	double* nerv_intra = (double*)mxMalloc(ndata * sizeof(double));
	double* nerv_inter = (double*)mxMalloc(ndata * sizeof(double));
	// Create arrays for output neighbor probabilities, log-probabilities
	// and gradient.
	double *distRowMin=(double*)mxMalloc(ndata*sizeof(double)); //minimum distance to each point.
	double *distClassRowMin=(double*)mxMalloc(ndata*nclasses*sizeof(double)); //  minimum distance in each sensitive class to each point.  
	double* Ydist = (double*)mxMalloc(ndata * ndata * sizeof(double)); // distance matrix of Y
	double *logExpSum=(double*)mxMalloc(ndata*sizeof(double));
	double *ExpClassSum=(double*)mxMalloc(ndata*nclasses*sizeof(double));
	double* P = (double*)mxMalloc(ndata * ndata * sizeof(double));
	double* logP = (double*)mxMalloc(ndata * ndata * sizeof(double));
    double *Q = (double *)mxMalloc(ndata*ndata*sizeof(double));
    double *logQ = (double *)mxMalloc(ndata*ndata*sizeof(double));
    double *grads = (double *)mxMalloc(ndata*ndim*sizeof(double));
	double* adpRatechange = (double*)mxMalloc(ndata * ndim * sizeof(double));
	double* Ychange = (double*)mxMalloc(ndata * ndim * sizeof(double));
	for (int i = 0; i < ndata * ndim; i++) adpRatechange[i] = 1.0;
	for (int i = 0; i < ndata * ndim; i++) Ychange[i] = 0;
	
    
	// early exaggeration
	for (int i = 0; i < ndata; i++)
	{
		for (int j = 0; j < ndata; j++)
		{
			P[j * ndata + i] = max(input_P[j * ndata + i] * exaggeration, DBL_MIN);
		}		
	}
	for (int i = 0; i < ndata * ndata; i++)
	{
		logP[i] = log(P[i]);
	}

    // Main loop of optimization
    for (iter = 0; iter < niters; iter++)
	{   
		if (iter == exaggerationStop)
		{
			for (int i = 0; i < ndata; i++)
			{
				for (int j = 0; j < ndata; j++)
				{
					P[j * ndata + i] = max(P[j * ndata + i] / exaggeration, DBL_MIN);
					
				}				
			}
			for (int i = 0; i < ndata * ndata; i++)
			{
				logP[i] = log(P[i]);
			}
		}
		
		
		if (iter == momentumChange) { momentum = final_momentum; }
	    // Compute output neighbor probabilities
		compute_neighborprob(ndata, ndim,nclasses,classes, Y, Ydist,distRowMin,distClassRowMin,logExpSum,ExpClassSum,sigma2, Q, logQ);
		
	    // Compute cost
        //compute_totalcost(ndata,ndim,P,Q,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,tradeoff_intra,tradeoff_inter,beta,gamma,&cost,&reccost_intra,&preccost_intra,&reccost_inter,&preccost_inter,&faircost1,&faircost2);
	    // Compute gradient
	    compute_grads(ndata,ndim,Y,Ydist,distRowMin,distClassRowMin,logExpSum,ExpClassSum,P,Q,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,temp_sumterms,sigma2,tradeoff_intra,tradeoff_inter,beta,gamma,nerv_intra,nerv_inter,grads);
	    // Compute squared norm of the gradient for tracking convergence
        double gradientnorm = 0;
	    for (i=ndata*ndim-1; i >= 0; i--)
	    {
			gradientnorm += grads[i]*grads[i];
			
		}
			    		

		// If gradient is overly large, normalize it to avoid wild jumps
		if (gradientnorm > 1)
		{
			gradientnorm = sqrt(gradientnorm);
	        for (i=ndata*ndim-1; i >= 0; i--)
	        {
			    grads[i] /= gradientnorm;
		    }			
		}
		
		
		// Update adpRatechange
		for (int i = 0; i < ndata * ndim; i++) adpRatechange[i] = (sign(grads[i]) != sign(Ychange[i]))
			? (adpRatechange[i] + .15) : (adpRatechange[i] * .85);
		for (int i = 0; i < ndata * ndim; i++) if (adpRatechange[i] < .01) adpRatechange[i] = .01;

		// Update Y
		for (int i = 0; i < ndata * ndim; i++) Ychange[i] = momentum * Ychange[i] -learning_rate*adpRatechange[i] * grads[i];

		//for (int i = 0; i < ndata * ndim; i++) Ychange[i] = momentum * Ychange[i] - learning_rate * grads[i];
		for (int i = 0; i < ndata * ndim; i++) Y[i] = Y[i] + Ychange[i];

        
		
		if (isnan(cost))
		{
		    break;
		}

		zeroMean(Y, Y_mean, ndata, ndim);
		
	}
	
    // Free temporary arrays
	mxFree(P);
	mxFree(logP);
	mxFree(distRowMin);
	mxFree(distClassRowMin);
	mxFree(Ydist);
	mxFree(logExpSum);
	mxFree(ExpClassSum);
    mxFree(Q);
    mxFree(logQ);
    mxFree(grads);	
	mxFree(overallclassprobs);
	mxFree(desiredclassprobs);
	mxFree(classprobs);
	mxFree(Y_mean);
	mxFree(temp_sumterms);
	mxFree(Ychange);
	mxFree(adpRatechange);
	mxFree(nerv_intra);
	mxFree(nerv_inter);
}

double get_mexdoublevalue
(
    const mxArray *mex_matrix
)
{
    double *matrix_data = (double *)mxGetData(mex_matrix);
    double doublevalue = matrix_data[0];
	return(doublevalue);
}

double *get_mexmatrixdata
(
    const mxArray *mex_matrix, 
	int *output_nrows, 
	int *output_ncols
)
{
	if ((output_nrows != NULL) && (output_ncols != NULL))
	{
        mwSize *matrix_dims = (mwSize *)mxGetDimensions(mex_matrix);
        *output_nrows = matrix_dims[0];
		*output_ncols = matrix_dims[1];
	}
    double *output_matrix_data = (double *)mxGetData(mex_matrix);
	return(output_matrix_data);
}



void mexFunction
(
    int nlhs,              // Number of left-hand side arguments
    mxArray *plhs[],       // Left-hand side arguments 
    int nrhs,              // Number of right-hand side arguments
    const mxArray *prhs[]  // Right-hand side arguments
)
{
    int ndata;            // Number of data points
	int ndim;             // Number of dimensions
	
    //mexPrintf ("Beginning mex function\n");

    // Access input parameters and create output parameters
    //mexPrintf ("Accessing and creating matrices\n");

    // Input: neighborhood probability.
	double *P_data = get_mexmatrixdata(prhs[0], NULL, NULL);
    // Input: number of classes of sensitive attribute
	int nclasses = get_mexdoublevalue(prhs[1]);
    // Input: class labels of sensitive attribute
	double *classes_data = get_mexmatrixdata(prhs[2], NULL, NULL);
    // Input: initial output coordinates
    double *Yinitial_data = get_mexmatrixdata(prhs[3], &ndata, &ndim);
    // Input: neighborhood squared widths sigma2
	double *sigma2_data = get_mexmatrixdata(prhs[4], NULL, NULL);
    // Input: classNeRV tradeoff parameter  within classes tradeoff_intra
	double tradeoff_intra = get_mexdoublevalue(prhs[5]);
	// Input: classNeRV  tradeoff parameter between classes tradeoff_inter
	double tradeoff_inter = get_mexdoublevalue(prhs[6]);
    // Input: fairness tradeoff parameter beta
	double beta = get_mexdoublevalue(prhs[7]);
	// Input: tradeoff parameter gamma
	double gamma = get_mexdoublevalue(prhs[8]);
    // Input: number of iterations
	int niters = get_mexdoublevalue(prhs[9]);
    // Input: learning rate
	double learning_rate = get_mexdoublevalue(prhs[10]);
    // Input: momentum multiplier
	double exaggeration = get_mexdoublevalue(prhs[11]);
	double contrast_coeffi = get_mexdoublevalue(prhs[12]);
	
    // Output: output coordinate matrix
    mxArray *mex_Y;
    mwSize *Y_dims;
    double *Y_data;
    Y_dims = (mwSize *)mxMalloc(2*sizeof(mwSize));
    Y_dims[0] = ndata;
    Y_dims[1] = ndim;
    mex_Y = mxCreateNumericArray(2, Y_dims, mxDOUBLE_CLASS, mxREAL);
    Y_data = (double *)mxGetData(mex_Y);
    plhs[0] = mex_Y;
    mxFree(Y_dims);

    // Perform the computation
   // mexPrintf ("Performing the computation\n");

    optimize_embedding(ndata, ndim, P_data, nclasses, classes_data, Yinitial_data, Y_data, sigma2_data, tradeoff_intra,tradeoff_inter, beta,gamma,contrast_coeffi, niters, learning_rate, exaggeration);

    //mexPrintf ("End of mex function\n");
}
