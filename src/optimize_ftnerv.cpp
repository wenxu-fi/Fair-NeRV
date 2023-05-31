// This is the implementation of the Fair-t-NeRV algorithm.
//In our algorithm, we have data with categorical sensitive attribute. Before using this you should preprocess data so that minimum sensitive attribute value is 0
#include <stdlib.h>
#include <math.h>
//#include <limits.h>
#include <float.h>
#include "mex.h"
//#include <algorithm>
#define MINCLASS 0
#define max(x,y) ((x)>(y)?(x):(y))

#define MATRIXELEMENT(ROW,COL,NROWS,NCOLS) (COL*NROWS+ROW)

static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


void compute_Euclidean_distance
(
    int ndata, 
    int ndim, 
    double *X, 
    double *Xdist // Xdist is 1+d2 in fact
)
{
    int i, j, k;
    double myinf=1e+16;
    // Compute Euclidean distance
    for (i = ndata-1; i >= 0; i--)
    {		
 	    for (j = i-1; j >= 0; j--) 
	    {
		    double tempdist = 0;			
		    for (k = (ndim-1)*ndata; k >= 0; k-=ndata)
		    {
			    double tempdiff = X[k+i]-X[k+j];
			    tempdist += tempdiff*tempdiff;
		    }
			// Fill in distance value and its symmetric pair
		    Xdist[j*ndata+i] = 1+tempdist;
		    Xdist[i*ndata+j] =1+ tempdist;
			
	    }
       // Xdist[i*ndata+i] = 0;
		Xdist[i * ndata + i] = myinf;
     }
    
	// Print distances for debugging
	   // mexPrintf("1/ii i %d %f\n",i, 1/Xdist[0*ndata+0]);
	    

    
  
}


// Compute t-distributed NeRV neighbor probability
void compute_neighborprob_t
(
    int ndata, 
    int nclasses,
    double *classes,
    double *D, 
    double *DRowMin,
    double *DRowClassMin,
    double *WClassSum, 
    double *WSum,
    double *Q,
    double *logQ  
)
{
	int i,j;
	double myinf=1e+16;

    for (int i=ndata*nclasses-1;i>=0;i--)
    {
		DRowClassMin[i]=myinf;
		WClassSum[i]=0;
	}
	

	 for (i=ndata-1;i>=0;i--)
	 {
		 DRowMin[i]=myinf;
		 WSum[i]=0;
		 for (j = ndata-1; j >= 0; j--)
		 {
			 int s_j=(int)classes[j]-MINCLASS;
			 int i_sj=s_j*ndata+i;
			 int i_j=j*ndata+i;
			 if (D[i_j]<DRowMin[i])
			 {
				 DRowMin[i]=D[i_j];
				 
			 }
			 if (D[i_j]<DRowClassMin[i_sj])
			 {
				 DRowClassMin[i_sj]=D[i_j];
				 
			 }			 
	     }
	     
	     for (j=ndata-1;j>=0;j--)
	     {   
			 int s_j=(int)classes[j]-MINCLASS;
			 WClassSum[s_j*ndata+i]+=DRowClassMin[s_j*ndata+i]/D[j*ndata+i];
			 double temp=DRowMin[i]/D[j*ndata+i];
			 Q[j*ndata+i]=temp;
			 WSum[i]+=temp;
		 }	
		 for (j=ndata-1;j>=0;j--)
		 {
			 Q[j*ndata+i] /=WSum[i];
			 logQ[j*ndata+i]=log(DRowMin[i])-log(D[j*ndata+i])-log(WSum[i]);
	     }
	     Q[i*ndata+i]=0;
	     logQ[i*ndata+i]=-DBL_MAX;	 
	 }
}

// compute Fair-t-NeRV cost function
	void compute_ftnerv_totalcost
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
		double tradeoff_intra,           //tradeoff of recall and precision in the same class 
		double tradeoff_inter,     // tradeoff of recall and precision between classes
		double beta,             // Fairness vs NeRV cost balance parameter
		double gamma,           
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
		double cost;
		double reccost_intra, preccost_intra, reccost_inter, preccost_inter;
		double faircost, faircost1, faircost2;

		// Compute Nerv cost: recall and precision KL divergences
		reccost_intra =preccost_intra=reccost_inter=preccost_inter= 0;
		
		for (i = ndata-1 ; i >= 0; i--)
		{
			for (j = ndata-1; j >= 0; j--)
			{   
				int tempofs_jndatai = j * ndata + i;
				if (classes[i] == classes[j])
				{
					reccost_intra += P[tempofs_jndatai] * (logP[tempofs_jndatai] - logQ[tempofs_jndatai])+Q[tempofs_jndatai]-P[tempofs_jndatai];
					preccost_intra += Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai])+P[tempofs_jndatai]-Q[tempofs_jndatai];
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
		
		// Compute fairness cost, faircost1: KL(desiredclassprobs,classprobs),faircost22: KL(classprobs,desiredclassprobs)
		faircost1 = 0;
		faircost2 = 0;
		for (i = ndata - 1; i >= 0; i--)
		{
			//mexPrintf ("Computing low dimension class distr for point %d\n", i);

			double* classprobs_i = classprobs + i;
			for (j = nclasses - 1; j >= 0; j--)
			{
				classprobs_i[j * ndata] = 0;
			}
			// Compute class distribution around point i
			for (j = ndata - 1; j >= 0; j--)
			{
				classprobs_i[((int)classes[j] - MINCLASS) * ndata] += Q[j * ndata + i];
			}
			
			//mexPrintf ("Computing class KL for point %d, %d classes\n", i, nclasses);
		//delay(1000);
			// Compute KL to overall class distribution
			double klvalue1 = 0;
			double klvalue2 = 0;
			for (j = nclasses - 1; j >= 0; j--)
			{
				//mexPrintf ("Computing class KL for point %d class %d\n", i, j);
				//klvalue1 += desiredclassprobs[j*ndata+i]*(log(desiredclassprobs[j*ndata+i])-log(classprobs_i[j*ndata]));
				klvalue1 += desiredclassprobs[j * ndata + i] * (log(max(desiredclassprobs[j * ndata + i], DBL_MIN)) - log(classprobs_i[j * ndata]));

				//klvalue2 += classprobs_i[j*ndata] * (log(classprobs_i[j*ndata]) - log(desiredclassprobs[j*ndata+i]));
				klvalue2 += classprobs_i[j * ndata] * (log(classprobs_i[j * ndata]) - log(max(desiredclassprobs[j * ndata + i], DBL_MIN)));

				//mexPrintf("kl(RU), %f\n", klvalue);
			   //mexPrintf("i %d,j %d,classprobs %f desiredclassprobs %f\n",i,j, classprobs_i[j*ndata],desiredclassprobs[j*ndata+i]); 
			}
			faircost1 += klvalue1; 
			faircost2 += klvalue2;
		}
		faircost = gamma * faircost1 + (1 - gamma) * faircost2;
		
		// Compute total cost
		cost = beta * (tradeoff_intra* reccost_intra + (1 - tradeoff_intra) * preccost_intra
			+tradeoff_inter*reccost_inter+(1-tradeoff_inter)*preccost_inter) + (1 - beta) * faircost;

		// Assign outputs
		*output_cost = cost;
		*output_reccost_intra = reccost_intra;
		*output_preccost_intra = preccost_intra;
		*output_reccost_inter = reccost_inter;
		*output_preccost_inter = preccost_inter;
		*output_faircost1 = faircost1;
		*output_faircost2 = faircost2;
	}


void compute_ftnerv_grads
(
    int ndata,
	int ndim,
	double *Y,
    double *P,
	double *Q,
	double *D,
	double *DRowMin,
	double *DRowClassMin,
	double *WClassSum,
	double *WSum,
	double *logP,
	double *logQ,
	int nclasses,
	double *classes,
	double *desiredclassprobs, 
	double *classprobs,        
	double *temp_sumterms,
	double tradeoff_intra,
	double tradeoff_inter,
	double beta,
	double gamma,
	double *tnerv_intra,
	double *tnerv_inter,
	double *grads
)
{
	int i, j, k;
	

	// Initialize gradient to zero
	for (i=ndata*ndim-1; i >= 0; i--)
	{
		grads[i] = 0;
	}

	//Initialize and compute  tnerv_intra and tnerv_inter
	for (i = ndata - 1; i >= 0; i--)
	{
		tnerv_intra[i] = tnerv_inter[i] = 0;
	}
	for (i = ndata - 1; i >= 0; i--)
	{
		for (j = ndata - 1; j >= 0; j--)
		{
			int tempofs_jndatai = j * ndata + i;
			if (classes[j] == classes[i])
			{
				tnerv_intra[i] += tradeoff_intra * (Q[tempofs_jndatai] - P[tempofs_jndatai]) +
					(1 - tradeoff_intra) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai]);
			}
			else
			{ 
				tnerv_inter[i] += tradeoff_inter * (Q[tempofs_jndatai] - P[tempofs_jndatai]) +
					(1 - tradeoff_inter) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai]);
			}
		}
	}
	
	// Compute class distributions for fairness gradient
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

	

	//compute array with element  D_kl(r_i,\rho_i)
	for (i = ndata - 1; i >= 0; i--)
	{   
		temp_sumterms[i] = 0;
		double *classprobs_i = classprobs + i;
		double* desiredclassprobs_i = desiredclassprobs + i;
		for (j = nclasses - 1; j >= 0; j--)
		{
			temp_sumterms[i] += classprobs_i[j * ndata] * (log(WClassSum[j*ndata+i])-log(DRowClassMin[j*ndata+i])
			+log(DRowMin[i])-log(WSum[i])-log(max(desiredclassprobs_i[j*ndata],DBL_MIN)));
			
			//temp_sumterms[i] += classprobs_i[j * ndata] * (log(classprobs_i[j * ndata]) - log(max(desiredclassprobs_i[j*ndata],DBL_MIN)));
			//mexPrintf("i %d %f\n", i, j, temp_sumterms[i]);
		}

	}
  //Gradient
	for (i=ndata-1; i >= 0; i--)
	{
		double *classprobs_i = classprobs+i;
		double* desiredclassprobs_i = desiredclassprobs + i;
		int s_i = (int)classes[i] - MINCLASS;
		for (j=ndata-1; j >= 0; j--)
		{ 
			int s_j = (int)classes[j] - MINCLASS;
			int tempofs_jndatai = j*ndata+i;
			int tempofs_indataj = i*ndata+j;
			//compute gradient of classnerv
			double temp_multiplier1 = tnerv_intra[i]*(1/D[tempofs_jndatai])*Q[tempofs_jndatai]
				+ tnerv_intra[j]*(1/D[tempofs_indataj])*Q[tempofs_indataj]
				+tnerv_inter[i] *(1/D[tempofs_jndatai])* Q[tempofs_jndatai]
				+ tnerv_inter[j] * (1/D[tempofs_indataj]) * Q[tempofs_indataj];
			if (classes[j] == classes[i])
			{
				temp_multiplier1 -=  (tradeoff_intra * (Q[tempofs_jndatai] - P[tempofs_jndatai])
					+ (1 - tradeoff_intra) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai])) * (1/D[tempofs_jndatai])
					+  (tradeoff_intra * (Q[tempofs_indataj] - P[tempofs_indataj])
						+ (1 - tradeoff_intra) * Q[tempofs_indataj] * (logQ[tempofs_indataj] - logP[tempofs_indataj])) * (1/D[tempofs_indataj]);
			}else
			{temp_multiplier1-= (tradeoff_inter * (Q[tempofs_jndatai] - P[tempofs_jndatai])
						+ (1 - tradeoff_inter) * Q[tempofs_jndatai] * (logQ[tempofs_jndatai] - logP[tempofs_jndatai])) * (1/D[tempofs_jndatai])
						+ (tradeoff_inter * (Q[tempofs_indataj] - P[tempofs_indataj])
							+ (1 - tradeoff_inter) * Q[tempofs_indataj] * (logQ[tempofs_indataj] - logP[tempofs_indataj])) * (1/D[tempofs_indataj]);}
			//compute gradient of fair part
			double temp_multiplier2=(gamma*(Q[tempofs_jndatai]-desiredclassprobs_i[s_j*ndata]*(DRowClassMin[s_j*ndata+i]/D[tempofs_jndatai])/WClassSum[s_j*ndata+i])
			+(1-gamma)*(log(WClassSum[s_j*ndata+i])-log(DRowClassMin[s_j*ndata+i])+log(DRowMin[i])-log(WSum[i])-log(max(desiredclassprobs_i[s_j*ndata],DBL_MIN)) - temp_sumterms[i])
			*Q[tempofs_jndatai])/D[tempofs_jndatai]+ (gamma*(Q[tempofs_indataj]-desiredclassprobs[s_i*ndata+j]*(DRowClassMin[s_i*ndata+j]/D[tempofs_indataj])/WClassSum[s_i*ndata+j])
			+(1-gamma)*(log(WClassSum[s_i*ndata+j])-log(DRowClassMin[s_i*ndata+j])+log(DRowMin[j])-log(WSum[j])
			-log(max(desiredclassprobs[s_i*ndata+j],DBL_MIN)) - temp_sumterms[j])*Q[tempofs_indataj])/D[tempofs_indataj];
			double *Y_rowj = Y+j;
			double *Y_rowi = Y+i;
			double *grads_rowi = grads+i;
		    for (k=(ndim-1)*ndata; k >= 0; k-=ndata)
		    {
			    grads_rowi[k] += (beta*temp_multiplier1-(1-beta)*temp_multiplier2)*(Y_rowi[k]-Y_rowj[k]);
		    }
		}
	}
    for (i=ndata*ndim-1; i >= 0; i--)
	{
		grads[i] *=2 ;
	}
   
}



void compute_empirical_gradient
(
    int ndata,
	int ndim,
    double *P, 
    double *Q,
    double *D,
    double *DRowMin,
    double *DRowClassMin,
    double *WClassSum,
    double *WSum,
	double *logP, 
	double *logQ, 
	int nclasses,
	double *classes,
	double *desiredclassprobs,
	double *classprobs,
	double *Y,
	double tradeoff_intra, 
	double tradeoff_inter,
	double beta,
	double gamma,
	double *grads
)
{
	int i, j, k;
	double cost, reccost_intra, preccost_intra,reccost_inter,preccost_inter, faircost1,faircost2;
	double cost0, reccost0_intra, preccost0_intra,reccost0_inter,preccost0_inter,faircost01,faircost02;
	
	// Compute base cost
    // Compute output neighbor probabilities
    compute_Euclidean_distance(ndata,ndim,Y,D);
 	compute_neighborprob_t(ndata, nclasses,classes,D, DRowMin,DRowClassMin,WClassSum, WSum,Q,logQ);		
			
	// Compute cost
    compute_ftnerv_totalcost(ndata,ndim,P,Q,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,tradeoff_intra,tradeoff_inter,beta,gamma,&cost0,&reccost0_intra,&preccost0_intra,&reccost0_inter,&preccost0_inter,&faircost01,&faircost02);
	//mexPrintf("cost0 %f (%f %f %f)\n",cost0,reccost0,preccost0,faircost01);
	double myepsilon = 1e-4;
	for (i=ndata-1; i >= 0; i--)
	{
		for (j=ndim-1; j >= 0; j--)
		{
			Y[j*ndata+i] += myepsilon;

	        // Compute output neighbor probabilities
	        compute_Euclidean_distance(ndata,ndim,Y,D);
	        compute_neighborprob_t(ndata, nclasses,classes,D, DRowMin,DRowClassMin,WClassSum, WSum,Q,logQ);	
 	        //compute_neighborprob_t(ndata, ndim, Y, Q, logQ, W);				
	        // Compute cost
            compute_ftnerv_totalcost(ndata,ndim,P,Q,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,tradeoff_intra,tradeoff_inter,beta,gamma,&cost,&reccost_intra,&preccost_intra,&reccost_inter,&preccost_inter,&faircost1,&faircost2);
			//mexPrintf("i%d,j%d,cost %f (%f %f %f)\n",i,j,cost,reccost,preccost,faircost1);
			grads[j*ndata+i] = (cost-cost0)/myepsilon;

			Y[j*ndata+i] -= myepsilon;
		}
	}
}


void gradientcheck
(
    int ndata,
	int ndim,
    double *P, 
    double *Q, 
    double *D,
    double *DRowMin,
    double *DRowClassMin,
    double *WClassSum,
    double *WSum,
	double *logP, 
	double *logQ, 
	int nclasses,
	double *classes,
	double *desiredclassprobs,
	double *classprobs,
	double *temp_sumterms,
	double *tnerv_intra,
	double *tnerv_inter,
	double *Y,
	double tradeoff_intra,
	double tradeoff_inter,
	double beta,
	double gamma,
	double *grads
)
{
	int i, j, k;
	
	// Compute output neighbor probabilities
	compute_Euclidean_distance(ndata,ndim,Y,D);
 	compute_neighborprob_t(ndata, nclasses,classes,D, DRowMin,DRowClassMin,WClassSum, WSum,Q,logQ);		
	// Compute analytical gradient
	//compute_W_related(ndata,nclasses,classes,W,WRowMax,WRowClassMax,WClassSum,WSum);
	compute_ftnerv_grads(ndata,ndim,Y,P,Q,D,DRowMin,DRowClassMin,WClassSum,WSum,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,temp_sumterms,tradeoff_intra,tradeoff_inter,beta,gamma,tnerv_intra,tnerv_inter,grads);
    //mexPrintf("KLQPin gradient check, %f \n",klqp);
	// Print probabilities for debugging
	mexPrintf("Analytical gradient\n");
    for (i = 0; i <= 5; i++)
    {
		for (j = 0; j < 2; j++)
		{
			mexPrintf("%f, ", grads[j*ndata+i]);
		}
		mexPrintf("\n");
	}

	// Compute empirical gradient
	//compute_empirical_gradient(ndata,ndim,P,Q,D,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,Y,tradeoff_intra,tradeoff_inter,beta,gamma,grads);
    compute_empirical_gradient(ndata,ndim,P,Q,D,DRowMin,DRowClassMin,WClassSum,WSum,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,Y,tradeoff_intra,tradeoff_inter,beta,gamma,grads);
	// Print probabilities for debugging
	mexPrintf("Empirical gradient\n");
    for (i = 0; i <= 5; i++)
    {
		for (j = 0; j < 2; j++)
		{
			mexPrintf("%f, ", grads[j*ndata+i]);
		}
		mexPrintf("\n");
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
	for (int i = ndata - 1; i >= 0; i--)
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
			Y[j * ndata + i] -= Y_mean[j];
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
	double cost, reccost_intra, preccost_intra,reccost_inter,preccost_inter, faircost1,faircost2;
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
	

	// Create a zero-initialized difference term for momentum
	double* differenceterm = (double*)mxMalloc(ndata * ndim * sizeof(double));
	for (i = ndata * ndim - 1; i >= 0; i--)
	{
		differenceterm[i] = 0;
	}

    // Compute overallclass probabilities (u in paper)
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
	double* desiredclassprobs = (double*)mxMalloc(ndata * nclasses * sizeof(double));
	for (i = 0; i < ndata; i++)
	{
		int sens_i = (int)classes[i] - MINCLASS;
		//mexPrintf("i %d sens_i %d\n", i, sens_i);
		for (j = 0; j < nclasses; j++)
		{
			if (sens_i == j)
			{
				desiredclassprobs[j * ndata + i] = 1 - contrast_coeffi;

			}
			else { desiredclassprobs[j * ndata + i] = contrast_coeffi * overallclassprobs[j] / (1 - overallclassprobs[sens_i]); }
			//mexPrintf("i%d j %d %f ",i, j, desiredclassprobs[j * ndata + i]);
		}
		//mexPrintf("\n");
	}
	
    // Create a preallocated array for class probabilities (r_i in paper)

    double *classprobs = (double *)mxMalloc(ndata*nclasses*sizeof(double));
	double* Y_mean = (double*)mxMalloc(ndim * sizeof(double));

    

    // create two prelocated arrays for gradient of tnerv cost
    double *tnerv_intra = (double *)mxMalloc(ndata*sizeof(double));
	double *tnerv_inter = (double *)mxMalloc(ndata * sizeof(double));
   
	//create a preallocated array for the numerator of P,1+d2
	double *D = (double*)mxMalloc(ndata * ndata * sizeof(double));
	
	//max value for each row in W
	double *DRowMin=(double*)mxMalloc(ndata*sizeof(double));
	//maxvalue for each row and sensitive attribute value
	double *DRowClassMin=(double*)mxMalloc(ndata*nclasses*sizeof(double));
	double *WClassSum=(double*)mxMalloc(ndata*nclasses*sizeof(double));
	double *WSum=(double*)mxMalloc(ndata*sizeof(double));
	// Create arrays for output neighbor probabilities, log-probabilities
	// and gradient.
	double *P = (double *)mxMalloc(ndata*ndata*sizeof(double));
	double *logP = (double *)mxMalloc(ndata*ndata*sizeof(double));
    double *Q = (double *)mxMalloc(ndata*ndata*sizeof(double));
    double *logQ = (double *)mxMalloc(ndata*ndata*sizeof(double));
    double *grads = (double *)mxMalloc(ndata*ndim*sizeof(double));
	// Create a preallocated array for sum terms needed in fairness gradient
	double* temp_sumterms = (double*)mxMalloc(ndata * sizeof(double));

	
	double *adpRatechange = (double *)mxMalloc(ndata*ndim*sizeof(double));
	double * Ychange = (double *)mxMalloc(ndata*ndim*sizeof(double));
	for (int i = 0; i < ndata * ndim; i++) adpRatechange[i] = 1.0;
	for (int i = 0; i < ndata * ndim; i++) Ychange[i] = 0;
	/*
	compute_Euclidean_distance(ndata,ndim,Y,D);
	
	compute_neighborprob_t(ndata, nclasses,classes,D, DRowMin,DRowClassMin,WClassSum, WSum,Q,logQ);		

	
	gradientcheck(ndata,ndim,input_P,Q,D,DRowMin,DRowClassMin,WClassSum,WSum,input_logP, logQ, nclasses, classes, desiredclassprobs, classprobs, temp_sumterms,tnerv_intra,tnerv_inter, Y, tradeoff_intra,tradeoff_inter, beta, gamma, grads);
	
	*/
	
	// early exaggeration
	for (int i = 0; i < ndata * ndata; i++)
	{
		P[i] = max(input_P[i] * exaggeration, DBL_MIN);
		logP[i] = log(P[i]);
	
	}
	
	
    // Main loop of optimization
    for (iter = 0; iter < niters; iter++)
	{   
		// Stop exaggerating about P-values after a while and switch momentum
		if (iter == exaggerationStop)
		{
			for (int i = 0; i < ndata*ndata;i++)
			{ 
				P[i] = max(P[i] / exaggeration, DBL_MIN);
				logP[i] = log(P[i]);
			}
		}
		
		if (iter == momentumChange) momentum = final_momentum;
	    
		// Compute output neighbor probabilities
		compute_Euclidean_distance(ndata,ndim,Y,D);
        compute_neighborprob_t(ndata, nclasses,classes,D, DRowMin,DRowClassMin,WClassSum, WSum,Q,logQ);		

				
	    // Compute cost
        //compute_ftnerv_totalcost(ndata,ndim,P,Q,logP,logQ,nclasses,classes,desiredclassprobs,classprobs,tradeoff_intra,tradeoff_inter,beta,gamma,&cost,&reccost_intra,&preccost_intra,&reccost_inter, &preccost_inter,&faircost1,&faircost2);
	    
		// Compute gradient
         compute_ftnerv_grads(ndata, ndim, Y, P, Q, D,DRowMin,DRowClassMin,WClassSum,WSum, logP, logQ, nclasses, classes, desiredclassprobs, classprobs, temp_sumterms, tradeoff_intra, tradeoff_inter, beta, gamma, tnerv_intra, tnerv_inter, grads);


		// Update adpRatechange
		for (int i = 0; i < ndata * ndim; i++) adpRatechange[i] = (sign(grads[i]) != sign(Ychange[i]))
			? (adpRatechange[i] + .15) : (adpRatechange[i] * .85);
		for (int i = 0; i < ndata * ndim; i++) if (adpRatechange[i] < .01) adpRatechange[i] = .01;
      
		// Compute squared norm of the gradient for tracking convergence
		double gradientnorm = 0;
		for (i = 0; i < ndata * ndim; i++)
		{
			gradientnorm += grads[i] * grads[i];
		}

		// Print optimization status
		//mexPrintf("iter %d, cost %f (%f, %f, %f, %f, %f, %f), gradientnorm %f\n", iter, cost, reccost_intra, preccost_intra,reccost_inter,preccost_inter, faircost1, faircost2,gradientnorm);

		
		// If gradient is overly large, normalize it to avoid wild jumps
		if (gradientnorm > 1)
		{
			gradientnorm = sqrt(gradientnorm);
			for (i = 0; i < ndata * ndim; i++)
			{
				grads[i] /= gradientnorm;
			}
		}
		
		// Update Y
		for (int i = 0; i < ndata * ndim; i++) Ychange[i] = momentum * Ychange[i] -learning_rate*adpRatechange[i] * grads[i];

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
	mxFree(differenceterm);
    mxFree(Q);
    mxFree(D);
    mxFree(DRowMin);
    mxFree(DRowClassMin);
    mxFree(WClassSum);
    mxFree(WSum);
    mxFree(logQ);
    mxFree(grads);	
	mxFree(overallclassprobs);
	mxFree(desiredclassprobs);
	mxFree(classprobs);
	mxFree(Y_mean);
	mxFree(temp_sumterms);
	mxFree(Ychange);
	mxFree(adpRatechange);
	mxFree(tnerv_intra);
	mxFree(tnerv_inter);
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

    // Input: neighborhood probability 
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
	//Input: contrast coefficient
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
    //mexPrintf ("Performing the computation\n");

    optimize_embedding(ndata, ndim, P_data, nclasses, classes_data, Yinitial_data, Y_data, tradeoff_intra,tradeoff_inter, beta,gamma,contrast_coeffi, niters, learning_rate, exaggeration);

    //mexPrintf ("End of mex function\n");
}
