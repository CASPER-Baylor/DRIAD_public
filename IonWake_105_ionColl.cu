/*
* Project: IonWake
* File Type: function library implementation
* File Name: IonWake_105_ionCollisions.cu
*
* Created: 6/15/2018
*
* Description:
*	Includes functions for calculating ion-neutral collisions.  
*	All routines are based on:
*	Particle-In-Cell + Monte-Carlo-Collision for Helium gas  (Peter Hartmann) 
* 	translated from he_pic_2009_f.cc Donko Zoltan   
*
* Functions:
*	setIonCrossSection_105()
*	ionCollisions_105()
*	setCollisonList_105()
* Local functions:
*   collisionIonNeutral
*	random_maxwell_velocity
*   errorFn_Inv
*
*/

// header file
#include "IonWake_105_ionColl.h"

/*
* Name: setIonCrossSections_105
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Reads in ion neutral collision cross sections and determines
*	total cross section and total ion collision frequency.
*
* Input:
*	gasType: type of gas (Argon or Neon)
*	i_cs_ranges: number of entries in range of ion cross sections
*
* Output (void):
*	sigma_i1: iotropic cross sections as a function of energy
*	sigma_i2: charge transfer cross sections as a function of energy
*	sigma_i_tot: total collision cross sections
*	tot_ion_coll_freq: summed ion collision frequency
*
*/
void setIonCrossSection_105
       (int gasType, 
		int i_cs_ranges,
		float NUM_DEN_GAS,
		float MASS_SINGLE_ION,
        float* sigma_i1, 
        float* sigma_i2, 
        float* sigma_i_tot, 
        float* tot_ion_coll_freq,
		const bool debugMode,
		OFile& fileName) {
		 
	const float depsilon_i 	= 0.001;
	const float ev_to_j = 1.602e-19;
	int j;
	float en;
	int n = 18;
	float en_v[n]; 
	float cs1_v[n]; 
	float cs2_v[n];
  
  	if (gasType == 1) {
	  //Ne cross sections for isotropic scatt. (cs1) and charge transfer (cs2)
	  //Ref: Jovanovic, Vrhovac, Petrovic, Eur. Phys. J. D 21, 335-342 (2002) 
	  //energy in eV
	  fileName << "Using collision cross sections for Neon \n";
	  float en_vNe[] = {0.001, 0.0032, 0.01,0.0145,0.02,0.032,0.05,0.08,0.13,
		0.2,0.32,0.5,0.8,1.25,2.0,10.0,100.0,1000.0};
		//cross sections in m^2
	  float cs1_vNe[] = {562,275,117,91.2,72.4,53.7,40.7,32.4, 26.9, 22.9,
		19.5, 16.2, 13.5, 11.5, 9.77, 5.34, 2.25, 0.949};
	  float cs2_vNe[] = {18.0, 19.0, 20.0, 21.4, 22.9, 25.7, 28.2, 29.5, 29.8,
		29.5, 28.8, 28.2, 27.5, 26.9, 27.5, 28.0, 28.5, 29.0};
	  for (int i=0;i<=n;i++){
			en_v[i] = en_vNe[i];
			cs1_v[i] = 1e-20 * cs1_vNe[i];
			cs2_v[i] = 1e-20 * cs2_vNe[i];
  	  } 
  	} else if (gasType == 2) {
			fileName << "Argon coming soon \n";
  	} else { 
			fileName << "Unknown gas type \n";
  	}

	//fileName << "----- Collision Cross Sections -----" << std::endl;
	for (int i = 0; i < n; i++) {
		fileName << en_v[i] << ", " << cs1_v[i] << ", "
			<< cs2_v[i] << '\n';
	}
	
	// Interpolate for fine divisions in energy scale
    for (int i=0;i<=i_cs_ranges;i++){
         if (i>0) en = depsilon_i*i; else en = depsilon_i;
         if (en<en_v[0]) {
			sigma_i1[i] = 0; 
			sigma_i2[i] = 0;
        } else if (en>en_v[n-1]) {
			sigma_i1[i] = 0; 
			sigma_i2[i] = 0;
        } else if (en==en_v[0]) {
			sigma_i1[i] = cs1_v[0]; 
			sigma_i2[i] = cs2_v[0];
        } else if (en==en_v[n-1]) {
				sigma_i1[i] = cs1_v[n]; 
				sigma_i2[i] = cs2_v[n];
		 } else { // linear interpolation
             j=1;
             while (en>=en_v[j]) {j=j+1;} 
			 j=j-1;
             sigma_i1[i]=cs1_v[j]+
				(cs1_v[j+1]-cs1_v[j])*(en-en_v[j])/(en_v[j+1]-en_v[j]); 
             sigma_i2[i]=cs2_v[j]+
				(cs2_v[j+1]-cs2_v[j])*(en-en_v[j])/(en_v[j+1]-en_v[j]);  
         }       
    }
 
	//  Sum the backscattering and isotropic scattering to get total  
	// ion impact cross section: 
  	for(int i=0;i<=i_cs_ranges;i++){
    	sigma_i_tot[i] = sigma_i1[i] + sigma_i2[i];
    	sigma_i_tot[i] *= NUM_DEN_GAS;
  	}  
 
	//Upper limit for collision frequency  
  	double   e,v,nu,nu_max;
  	nu_max = 0;
  	for(int i=1;i<=i_cs_ranges;i++){
    	e  = i*depsilon_i*ev_to_j;
    	v  = sqrt(e/MASS_SINGLE_ION);     
    	nu = v*sigma_i_tot[i];
    	if (nu > nu_max) {nu_max = nu;}
  	}

  	*tot_ion_coll_freq = nu_max; 

	if (debugMode) {
		fileName << "--- 1st 20 Interpolated Cross Sections ---" << '\n';
		for (int i = 0; i < 20; i++) {
			fileName << sigma_i1[i] << ", " << sigma_i2[i] << ", "
				<< sigma_i_tot[i] << '\n';
		}
		fileName << "--- Last 20 Interpolated Cross Sections ---" << '\n';
		for (int i = (i_cs_ranges - 20); i < i_cs_ranges; i++) {
			fileName << sigma_i1[i] << ", " << sigma_i2[i] << ", "
				<< sigma_i_tot[i] << '\n';
		}
	}
} 


/*
* Name: ionCollisions_105
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Performs collisions between ions and neutrals using null collision method.  
*	Particle-In-Cell + Monte-Carlo-Collision for Helium gas  (Peter Hartmann) 
* 	translated from he_pic_2009_f.cc Donko Zoltan             
*
* Input:
*	NUM_ION
* 	tot_ion_coll_freq
*	TIME_STEP = dt_i: ion time step
*	velIon
*	TEMP_ION
*	MASS_SINGLE_ION
*	sigma_i1: isotropic scattering cross section
*	sigma_i2: charge exchange cross section
*	sigma_i_tot: total cross section at energy eV
*
* Output (void):
*	velIon
*
*/

__global__ void ionCollisions_105 
	(int* d_collList,
	float* const d_TEMP_ION,
	float* const d_MASS_SINGLE_ION,
	float* const d_BOLTZMANN,
	int* const i_cs_ranges,
	float* const tot_ion_coll_freq,
	float* sigma_i1,
	float* sigma_i2,
	float* sigma_i_tot,
	float3* velIon,
	curandState_t* const randStates,
	int* d_collision_counter) {
			
	// thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;		
			
    // local variables
	const float ev_to_j 	= 1.602e-19;
	const float depsilon_i 	= 0.001; //cross section energy increment
	int			index;
	float	vx_i, vy_i, vz_i, vx_a, vy_a,vz_a;
	float	eps_rel, real_coll_freq, t1, t2, dum, randNum;
	double  khi,phi,mm;
	double  hx,hy,hz,g,gx,gy,gz,sk,ck,sf,cf;
	double  pi = 3.1415926536;
	*d_collision_counter = 0;

	if (d_collList[threadID] == 1) {
      //i = d_collList[threadID];
      vx_i = velIon[threadID].x;
      vy_i = velIon[threadID].y;
      vz_i = velIon[threadID].z;
      
      // select random maxwellian target: 
	  vx_a = __fsqrt_rn(-1.0);
	  while( isnan(vx_a) ) {
		// get a random number from -1 to 1
		randNum = curand_uniform(&randStates[threadID]) * 2.0 - 1.0;
  		vx_a = errorFn_inv(randNum);
	  }
	  vy_a = __fsqrt_rn(-1.0);
	  while( isnan(vy_a) ) {
		// get a random number from -1 to 1
		randNum = curand_uniform(&randStates[threadID]) * 2.0 - 1.0;
  		vy_a = errorFn_inv(randNum);
	  }
	  vz_a = __fsqrt_rn(-1.0);
	  while( isnan(vz_a) ) {
		// get a random number from -1 to 1
		randNum = curand_uniform(&randStates[threadID]) * 2.0 - 1.0;
  		vz_a = errorFn_inv(randNum);
	  }
	  dum = sqrt(2.0 * *d_BOLTZMANN * *d_TEMP_ION/ *d_MASS_SINGLE_ION);
	  vx_a *= dum;
	  vy_a *= dum;
	  vz_a *= dum;

      gx = vx_a-vx_i;
      gy = vy_a-vy_i;
      gz = vz_a-vz_i;
  
      g = sqrt(gx*gx + gy*gy + gz*gz);
	  //energy = 0.5*reduced_mass*v^2
      eps_rel = *d_MASS_SINGLE_ION * g * g /4.0/ev_to_j;  
      index = (int)(eps_rel/depsilon_i +0.5);

      if (index >= *i_cs_ranges) {index = *i_cs_ranges -1;}
      real_coll_freq = sigma_i_tot[index]*g;


	  randNum = curand_uniform(&randStates[threadID]);
      if (randNum < (real_coll_freq / *tot_ion_coll_freq)){
	  	t1  =     sigma_i1[index];
	  	t2  = t1 +sigma_i2[index];
	  	randNum = curand_uniform(&randStates[threadID]);
		if  (randNum < (t1 /t2)){
	  		randNum = curand_uniform(&randStates[threadID]);
			khi = acos(1.0-2.0*randNum); 
			//icollcounter1++;
		} else {
			khi = pi;
			//icollcounter2++;
		}
	  	randNum = curand_uniform(&randStates[threadID]);
		phi = 2.0*pi*randNum;
		sk  = sin(khi);
		ck  = cos(khi);
		sf  = sin(phi);
		cf  = cos(phi);
		mm  = sqrt(g*g-gx*gx);
		hx  = mm*cf;
		hy  = -(gx*gy*cf+g*gz*sf)/mm;
		hz  = -(gx*gz*cf-g*gy*sf)/mm;
		vx_i += 0.5*(gx*(1.0-ck)+hx*sk);
		vy_i += 0.5*(gy*(1.0-ck)+hy*sk);  
		vz_i += 0.5*(gz*(1.0-ck)+hz*sk);
		velIon[threadID].x = vx_i;
		velIon[threadID].y = vy_i;
		velIon[threadID].z = vz_i;
		++*d_collision_counter;
      }  
 
    }

}	

/*
* Name: setCollisionList_105()
*
* Description:
*	Zeros the collision list
*
* Inputs:
*	d_collList: list of ions to undergo collision
*
* Output (void):
*	d_collList: entries reset to -1
*
* Assumptions:
*	Number of entries = NUM_ION = multiple of block size
*
* Includes:
*	cuda_runtime.h
*	device_)launch_parameters.h
*
*/

__global__ void setCollisionList_105
	(int* d_collList, int value) {

	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	d_collList[threadID] = value;
}

/*
* Name: collisionIonNeutral
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Models a collision between ion and neutral atom, setting the new
*	velocity for the ion.
*
* Input:
*	index: the index of the ion which was chosen for collision
*	ionVel: velocity of ion before collision
*	MASS_SINGLE_ION
*	vx_2: velocity of neutral atom (squared)
*	vy_2:
*	vz_2:
*	t1: isotropic cross section
*	t2: charge exchange cross section
*
* Output (void):
*	ionVel: velocity of ion after collision
*
*/
/*
*void collisionIonNeutral(
*	int index, 
*	float3* velIon,  
*	double MASS_SINGLE_ION,
*	double vx_2, 
*	double vy_2, 
*	double vz_2,
*	double t1,
*	double t2){
*		
*  double  rnd,khi,phi,mm,vx_1,vy_1,vz_1;
*  double  hx,hy,hz,g,gx,gy,gz,sk,ck,sf,cf;
*  double  pi = 3.1415926536;
*  
*  vx_1 = velIon[index].x;
*  vy_1 = velIon[index].y;
*  vz_1 = velIon[index].z;
*
*  // random maxwellian target already selected (comes with the call)
*
*  // calculate relative velocity before begin collision:   
*
*  gx = vx_2-vx_1;
*  gy = vy_2-vy_1;
*  gz = vz_2-vz_1;
*  g  = sqrt(gx*gx+gy*gy+gz*gz);
*
*  rnd = (rand() % 100001)/100000.0;
*  if  (rnd < (t1 /t2)){
*    khi = acos(1.0-2.0*rand()); 
*    //icoll_counter[1]++;
*    //icollcounter1++;
*  } else {
*    khi = pi;
*    //icoll_counter[2]++;
*    //icollcounter2++;
*  }
*  phi = 2.0*pi*rand();
*  sk  = sin(khi);
*  ck  = cos(khi);
*  sf  = sin(phi);
*  cf  = cos(phi);
*  mm  = sqrt(g*g-gx*gx);
*  hx  = mm*cf;
*  hy  = -(gx*gy*cf+g*gz*sf)/mm;
*  hz  = -(gx*gz*cf-g*gy*sf)/mm;
*  //mm  = he_mass/(he_mass+he_mass);
*  vx_1 += 0.5*(gx*(1.0-ck)+hx*sk);
*  vy_1 += 0.5*(gy*(1.0-ck)+hy*sk);  
*  vz_1 += 0.5*(gz*(1.0-ck)+hz*sk);
*  velIon[index].x = vx_1;
*  velIon[index].y = vy_1;
*  velIon[index].z = vz_1;
*}
*/

//--------------------------------------------------------------------------    
// Maxwellian target sampling 
//--------------------------------------------------------------------------    
/*
* Name: errorFn_inv_105
* Created: 6/15/2018
*
* Editors
*	Name: Lorin Matthews
*	Contact: Lorin_Matthews@baylor.edu
*
* Description:
*	Inverts the error function -- used to create a random number with
*	a Gaussian probability distribution.  Algorithm adapted from
* 	Particle-In-Cell + Monte-Carlo-Collision for Helium gas (Peter Hartmann)  
* 	translated from he_pic_2009_f.cc Donko Zoltan             
*
* Input:
*	y: a random number -1 <   y    < 1
*
* Output (void):
*	x: -inf < erf^-1 < inf
*
*/

__device__ double errorFn_inv(double y) {
 
  double s, t, u, w, x, z; 
  double k = y;  // store y before switching its sign 
  
  if (y == 0) 
    { 
      x = 0; 
      return x; 
    } 
  if (y > 1.0) 
    { 
      x = -(__logf(0));       // to generate +inf 
      return x; 
    } 
  if (y < -1.0) 
    { 
      x = __logf(0);        // to generate -inf 
      return x; 
    } 
  if (y < 0) 
    y = -y;              // switch the sign of y it's negative 
                         // hence the comupation is for y >0 
  
  z = 1.0 - y; 
  w = 0.916461398268964 - __logf(z); 
  u = __fsqrt_rn(w); 
  s = (__logf(u) + 0.488826640273108) / w; 
  t = 1 / (u + 0.231729200323405); 
  x = u * (1.0 - s * (s * 0.124610454613712 + 0.5)) - 
    ((((-0.0728846765585675 * t + 0.269999308670029) * t + 
               0.150689047360223) * t + 0.116065025341614) * t + 
     0.499999303439796) * t; 
  t = 3.97886080735226 / (x + 3.97886080735226); 
  u = t - 0.5; 
  s = (((((((((0.00112648096188977922 * u + 
                           1.05739299623423047e-4) * u - 0.00351287146129100025) * u - 
             7.71708358954120939e-4) * u + 0.00685649426074558612) * u + 
           0.00339721910367775861) * u - 0.011274916933250487) * u - 
         0.0118598117047771104) * u + 0.0142961988697898018) * u + 
       0.0346494207789099922) * u + 0.00220995927012179067; 
  s = ((((((((((((s * u - 0.0743424357241784861) * u - 
                 0.105872177941595488) * u + 0.0147297938331485121) * u + 
               0.316847638520135944) * u + 0.713657635868730364) * u + 
             1.05375024970847138) * u + 1.21448730779995237) * u + 
           1.16374581931560831) * u + 0.956464974744799006) * u + 
         0.686265948274097816) * u + 0.434397492331430115) * u + 
       0.244044510593190935) * t - 
    z * __expf(x * x - 0.120782237635245222); 
  x += s * (x * s + 1.0); 
  
  if(k < 0) 
    return -x; 
  else 
    return x; 
} 

//--------------------------------------------------------------------
// sampling of Maxwellian distributions :
//
//double random_maxwell_velocity(void) {
//	double x;
//  	x = sqrt(-1.0);
//	while( isnan(x) ) {
//  		x = errorFn_inv((rand() % 2000 - 1000)/1000.0);
//	}
//	return x;
//}
