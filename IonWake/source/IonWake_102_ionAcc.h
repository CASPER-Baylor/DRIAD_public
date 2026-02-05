/*
 * Project: IonWake
 * File Type: function library implementation
 * File Name: IonWake_102_ionAcc.cu
 *
 * Created: 6/13/2017
 * Last Modified: 09/09/2020
 *
 * Description:
 *	Includes functions for calculating ion-ion accelerations
 *
 * Functions:
 *	calcIonAccels_102()
 *	calcIonIonAcc_102()
 *   calcIonDustAcc_102()
 *   calcIonDustAcc_102_dev()
 *   calcExtrnElcAcc_102()
 *
 * Includes:
 *	calcIonIonAcc_102()
 *		cuda_runtime.h
 *		device_launch_parameters.h
 *	calcIonDustAcc_102()
 *		cuda_runtime.h
 *		device_launch_parameters.h
 *	calcExtrnElcAcc_102()
 *		cuda_runtime.h
 *		device_launch_parameters.h
 *	calcExtrnElcAccCyl_102()
 *		cuda_runtime.h
 *		device_launch_parameters.h
 *	calcIonDensityPotential_102()
 *		cuda_runtime.h
 *		device_launch_parameters.h
 */

#ifndef IONWAKE_102_IONACC
#define IONWAKE_102_IONACC

/*
 * Required By:
 *	calcIonIonAcc_102()
 *	calcIonDustAcc_102()
 *	calcExtrnElcAcc_102()
 *	calcExtrnElcAccCyl_102()
 *	calcIonDensityPotential_102()
 * For:
 *	CUDA
 */
#include "cuda_runtime.h"

/*
 * Required By:
 *	calcIonIonAcc_102()
 *	calcIonDustAcc_102()
 *	calcExtrnElcAcc_102()
 *	calcExtrnElcAccCyl_102()
 *	calcIonDensityPotential_102()
 * For:
 *	CUDA
 */
#include "device_launch_parameters.h"

/*
 * Name: calcIonAccels_102
 * Created: 11/20/2020
 *
 * Editors
 *   Name: Lorin Matthews
 *   Contact: Lorin_Matthews@baylor.edu
 *
 * Description:
 *   Calculates the accelerations due to ion-ion
 *   interactions modeled as Yukawa particles, using both ions
 *   inside the simulation and force from ions outside simulation
 *   regtion (using table lookup).  Also add in acceleration from a
 *   constant external electric field (in z-direction).
 *
 * Input:
 *   d_posIon: the positions and charges of the ions
 *   d_accIon: the accelerations of the ions
 *   d_NUM_ION: the number of ions
 *   d_SOFT_RAD_SQRD: the squared softening radius squared
 *   d_ION_ION_ACC_MULT: a constant multiplier for the yakawa interaction
 *   d_INV_DEBYE: the inverse of the debye
 *   d_Q_DIV_M: charge to mass ratio of ions
 *   d_HT_CYL: half of the cylinder height
 *   d_Vout: table with potentials at points in rz-plane from outside ions
 *   d_NUMR: number of columns in table (radial direction)
 *   d_RESZ: number of rows in table (axial direction)
 *   d_dz: distance increment in z-direction
 *   d_dr: distance increment in radial direction
 *   d_E_FIELD: electric field in the z-direction
 *   d_E_FIELDR: radial electric field in plasma column
 *   E_direction: +/- z, for alternating DC E-field
 *   plasma_counter: used to increment evolving boundary conditions
 *   GEOMETRY: spherical or cylindrical simulation region
 *   d_EXTERN_ELC_MULT: used in calculating outside force for spherical boundary
 *   d_INV_DEBYE: used in calculation of outside force for spherical boundary
 *
 * Output (void):
 *   d_accIon: the acceleration due to all of the other ions
 *       is added to the initial ion acceleration
 *
 * Assumptions:
 *   All inputs are real values
 *   All ions have the parameters specified in the creation of the
 *       d_ION_ION_ACC_MULT value
 *   The ion potential is a Yukawa potential
 *   The number of ions is a multiple of the block size
 *
 * Includes:
 *   cuda_runtime.h
 *   device_launch_parameters.h
 *
 */

__global__ void calcIonAccels_102(float4 *, float4 *, float4 *,
                                  int *const, float *const,
                                  float *const, float *const,
                                  float *, float *const, float *,
                                  int *, int *, float *, float *,
                                  float *, float *, int,
                                  float *, float *, float *,
                                  int, int, float *const, int *const);

/*
 * calcIonIonAcc_102
 *
 * Editors
 *	Dustin Sanford
 *
 * Description:
 *	Calculates the accelerations due to ion-ion
 *	interactions modeled as Yakawa particles.
 *
 * Input:
 *	d_posIon: the positions and charges of the ions
 *	d_accIon: the accelerations of the ions
 *	d_NUM_ION: the number of ions
 *	d_SOFT_RAD_SQRD: the squared softening radius squared
 *	d_ION_ION_ACC_MULT: a constant multiplier for the yakawa interaction
 *	d_INV_DEBYE: the inverse of the debye
 *
 * Output (void):
 *	d_accIon: the acceleration due to all of the other ions
 *		is added to the initial ion acceleration
 *
 * Asumptions:
 *	All inputs are real values
 *	All ions have the parameters specified in the creation of the
 *		d_ION_ION_ACC_MULT value
 *   The ion potential is a Yukawa potential
 *   The number of ions is a multiple of the block size
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void calcIonIonAcc_102(float4 *, float4 *, int *const, float *const, float *const,
                                  float *const);

/*
 * calcIonDustAcc_102
 *
 * Editors
 *	Dustin Sanford
 *
 * Description:
 *	Calculates the ion accelerations due to ion-dust interactions
 *
 * Input:
 *	d_posIon: the positions of the ions, charge in 4th pos'n
 *	d_accIon: the accelerations of the ions
 *	d_posDust: the dust particle positions, charge in 4th pos'n
 *	d_NUM_ION: the number of ions
 *	d_NUM_DUST: the number of dust particles
 *	d_SOFT_RAD_SQRD: the squared softening radius squared
 *	d_ION_DUST_ACC_MULT: a constant multiplier for the ion-dust interaction
 *	d_INV_DEBYE: the inverse of the debye
 *   d_mindistDust: distance to closest dust particle
 *
 * Output (void):
 *	d_accIon: the acceleration due to all the dust particles
 *		is added to the initial ion acceleration
 *
 * Assumptions:
 *	All inputs are real values
 *	All ions and dust particles have the parameters specified in the
 *       creation of the d_ION_ION_ACC_MULT value
 *   The potential due to the dust particle is a bare coulomb potential
 *   The number of ions is a multiple of the block size
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void calcIonDustAcc_102(float4 *, float4 *, float4 *, int *const, int *const,
                                   float *const, float *const, float *);

/*
 * calcExtrnElcAcc_102
 *
 * Editors
 *	Dustin Sanford
 *
 * Description:
 *	calculates the acceleration on the ions due to the electric field
 *   created by the ions outside of a simulation sphere.
 *
 * Input:
 *	d_accIon: ion accelerations
 *	d_posIon: ion positions and charges
 *	d_EXTERN_ELC_MULT: constant multiplier for calculating the electric
 *       field due to the ions outside of the simulation sphere
 *	d_INV_DEBYE: the inverse debye length
 *
 * Output (void):
 *	d_accIon: the acceleration due to the outside electric
 *		field is added to the initial ion accelerations
 *
 * Assumptions:
 *	All inputs are real values
 *	The simulation region is a sphere
 *	The electric field due to outside ions is  radially symmetric
 *	All ions have the parameters specified in the creation of
 *		the d_EXTERN_ELC_MULT value
 *	The center of the simulation region is (0,0,0)
 *   The number of ions is a multiple of the block size
 *
 * Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void calcExtrnElcAcc_102(float4 *, float4 *, float *, float *);

/*
 * Name: calcExtrnElcAccCyl_102
 * Created: 11/18/2017
 * last edit: 10/13/2020
 *
 * Editors
 *       Name: Lorin Matthews
 *       Contact: Lorin_Matthews@baylor.edu
 *       last edit: 11/18/2017
 *
 * Description:
 *       Calculates the acceleration on the ions due to the electric field created
 *   by the ions outside of a simulation cylinder. This is implemented using a table
 *   lookup for the potential calculated for ions outside the cylinder. Note that the
 *   electric field is the gradient of the potential.  However, the potential of a
 *   cylindrical cavity is the negative of the potential of the cylinder of ions,
 *   which is what is calculated by boundaryEField. The two negatives cancel.
 *
 * Input:
 *       d_accIon: ion accelerations
 *       d_posIon: ion positions and charges
 *       d_Q_DIV_M:  charge to mass ratio
 * 		d_HT_CYL: half the cylinder height
 * 		d_ionOutPotential; potential of ions outside the simulation cylinder
 * 		d_NUMR: number of grid points in r-direction
 * 		d_dz: increment in z between grid points
 * 		d_dr: increment in r between grid points
 * 		d_Esheath: DC electric field in plasma column
 * 		E_dir: direction of electric field (and ion flow)
 * 		plasma_counter: index for evolving plasma conditions
 *
 *
 * Output (void):
 *       d_accIon: the acceleration due to the outside electric
 *               field is added to the initial ion accelerations
 *
 * Assumptions:
 *       All inputs are real values
 *       The simulation region is a cylinder
 *       The electric field due to outside ions is radially symmetric
 *       The center of the simulation region is (0,0,0)
 *       The coefficients for the electric fields were calculated using the
 *         Matlab routine e_field_in_cylinder.m using the correct dimentions
 *         for the cylinder, ion density, and debye length.
 *   The number of ions is a multiple of the block size
 *
 * Includes:
 *       cuda_runtime.h
 *       device_launch_parameters.h
 *
 *
 */
__global__ void calcExtrnElcAccCyl_102(float4 *, float4 *, float *, float *const, float *, int *,
                                       int *, float *, float *, float *, int, int);

/*
 *  Name: calcIonDensityPotential_102
 *  Created: 5/4/2018
 *  Last Modified:8.27.2020
 *
 *  Editors
 * 	Name: Lorin Matthews
 * 	Contact: Lorin_Matthews@baylor.edu
 * 	last edit: 9/10/2020
 * 	Implemented float2 for grid positions
 *
 *  Description:
 * 	Calculates electric potential from ions at points on grid in
 *  	the xz-plane.  Also calculates the number density at each grid
 * 	point by counting the number of ions in a sphere of radius r_dens
 *  	centered at each grid point.
 *
 *  Input:
 * 	d_posIion: ion positions
 * 	d_gridPos: the grid points in xz-plane
 * 	d_ION_POTENTIAL_MULT
 * 	d_INV_DEBYE
 *
 *  Output (void):
 * 	d_ionPotential: potential at each grid point
 * 	d_ionDenisty: ion number density at each grid point
 *
 *  Assumptions:
 *    The number of grid points is a multiple of the block size?????
 *
 *  Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void calcIonDensityPotential_102(const int, float2 *, float4 *,
                                            float *const, float *const,
                                            int *const, float *,
                                            float *);

/**
 * @brief calculate the ion density and potential on the 3D grid
 *
 * @param NUM_GRID_PTS_3D
 * @param d_gridPos3D
 * @param d_posIon
 * @param d_COULOMB_CONST
 * @param d_INV_DEBYE
 * @param d_NUM_ION
 * @param d_ionPotential3D
 * @param d_ionDensity3D
 * @return __global__
 */

__global__ void calcIonDensityPotential_3D_102(const int, float3 *, float4 *,
                                               float *const, float *const,
                                               int *const, float *,
                                               float *);

/*
 *  Name: zeroIonDensityPotential_102
 *  Created: 5/21/2018
 *  Last Modified: 5/21/2018
 *
 *  Editors
 * 	Name: Lorin Matthews
 * 	Contact: Lorin_Matthews@baylor.edu
 * 	last edit: 5/21/2018
 *
 *  Description:
 * 	Zeros electric potential from ions at points on grid in
 *  	the xz-plane.  Also zeros the number density at each grid
 *
 *  Input:
 * 	d_ionPotential: potential at each grid point
 * 	d_ionDenisty: ion number density at each grid point
 *
 *  Output (void):
 * 	d_ionPotential: potential at each grid point
 * 	d_ionDenisty: ion number density at each grid point
 *
 *  Assumptions:
 *    The number of grid points is a multiple of the block size
 *
 *  Includes:
 *	cuda_runtime.h
 *	device_launch_parameters.h
 *
 */
__global__ void zeroIonDensityPotential_102(int, float *, float *);
#endif
