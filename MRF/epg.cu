
#define STATES 2056

__device__ void transitionMatrix(float Tr[3][3], float Ti[3][3], float alpha, float phi)
{
    float halpha = 0.5*alpha;

    // real part 
    Tr[0][0] = cos(halpha)*cos(halpha);
    Tr[0][1] = sin(halpha)*sin(halpha)*cos(2.0*phi);// + 1j*np.sin(halpha)*np.sin(halpha)*np.sin(2*phi)
    Tr[0][2] = sin(alpha)*sin(phi);// - 1j*np.sin(alpha)*np.cos(phi)
    Tr[1][0] = sin(halpha)*sin(halpha)*cos(2.0*phi); //- 1j*np.sin(halpha)*np.sin(halpha)*np.sin(2.0*phi)
    Tr[1][1] = cos(halpha)*cos(halpha);
    Tr[1][2] = sin(alpha)*sin(phi);// + 1j*np.sin(alpha)*np.cos(phi)
    Tr[2][0] = -0.5*sin(alpha)*sin(phi);// - 1j*0.5*np.sin(alpha)*np.cos(phi)
    Tr[2][1] = -0.5*sin(alpha)*sin(phi);// + 1j*0.5*np.sin(alpha)*np.cos(phi)
    Tr[2][2] = cos(alpha);

    // imaginary part 
    Ti[0][0] = 0.0f;
    Ti[0][1] = sin(halpha)*sin(halpha)*sin(2*phi);
    Ti[0][2] = -sin(alpha)*cos(phi);
    Ti[1][0] = -sin(halpha)*sin(halpha)*sin(2.0*phi);
    Ti[1][1] = 0.0f;
    Ti[1][2] = sin(alpha)*cos(phi);
    Ti[2][0] = -0.5*sin(alpha)*cos(phi);
    Ti[2][1] = 0.5*sin(alpha)*cos(phi);
    Ti[2][2] = 0.0f;
}

__device__ void rf(float omr[3][STATES], float omi[3][STATES], float alpha, float phi, float Tr[3][3], float Ti[3][3], int nmax) 
{

    float xr[3], xi[3];

    // calculate the transition matrix 
    transitionMatrix(Tr, Ti, alpha, phi);

    // only apply to maximum number of configuration states 
    for (int i=0; i<nmax; i++) {

        for (int j=0; j<3; j++) { 
            xr[j] = 0.0f;
            xi[j] = 0.0f;
            for (int k=0; k<3; k++) { 
                xr[j] += (Tr[j][k]*omr[k][i] - Ti[j][k]*omi[k][i]);
                xi[j] += (Tr[j][k]*omi[k][i] + Ti[j][k]*omr[k][i]);
            }
        }
        for (int j=0; j<3; j++) {
            omr[j][i] = xr[j];
            omi[j][i] = xi[j];
        }
    }

}

__device__ void relax(float omr[3][STATES], float omi[3][STATES], float t1, float t2, float dt, int nmax)
{
    float E1 = exp(-dt/t1);
    float E2 = exp(-dt/t2);
    for (int n=0; n<nmax; n++) {
        omr[0][n] *= E2;
        omi[0][n] *= E2;
        omr[1][n] *= E2;
        omi[1][n] *= E2;
        omr[2][n] *= E1;
        omi[2][n] *= E1;
    }
    omr[2][0] += (1 - E1);
}   

__device__ void spoil(float omr[3][STATES], float omi[3][STATES], float omrcpy[3][STATES], float omicpy[3][STATES], int nmax)
{
    int i, j;
    for (i=0; i<2; i++) {
        for (j=0; j<nmax; j++) {
            omrcpy[i][j] = omr[i][j];
            omicpy[i][j] = omi[i][j];
        }
    }
    for (i=0; i<nmax; i++) {
        omr[0][i+1] = omrcpy[0][i];
        omi[0][i+1] = omicpy[0][i];
        omr[1][i] = omrcpy[1][i+1];
        omi[1][i] = omicpy[1][i+1];
    }
    omr[0][0] = omr[1][0];
    omi[0][0] = -omi[1][0];
}

__device__ void sample(float *yr, float *yi, float omr[3][STATES], float omi[3][STATES], float phi)
{
    *yr = omr[0][0]*cos(phi) - omi[0][0]*sin(phi);
    *yi = omr[0][0]*sin(phi) + omi[0][0]*cos(phi);
}



__global__ void gre_epg_cuda(float *destr, float *desti, float *alpha, float *phi, float *tr, float *t1, float *t2, int nrf, int nt1, int isIR)
{

    int n = blockIdx.x*blockDim.x + threadIdx.x;

    if ((n >= 0) && (n < nt1)) {

        // initialize magnetization 
        float omr[3][STATES];
        float omi[3][STATES];
        for (int i=0; i<3; i++) {
            for (int j=0; j<STATES; j++) {
                omr[i][j] = 0.0f;
                omi[i][j] = 0.0f;
            }
        }
        omr[2][0] = 1.0f;
        if (isIR) 
            omr[2][0] = -1.0f;

        // allocate another configuration matrix for fast spoiling operations 
        float omr_copy[3][STATES];
        float omi_copy[3][STATES];

        // real and imaginary part of RF pulse transition matrix 
        float Tr[3][3], Ti[3][3];

        // loop over RF pulses 
        for (int p=0; p<nrf; p++) {

            // apply the RF pulse 
            rf(omr, omi, alpha[p], phi[p], Tr, Ti, p+1);

            // sample the signal and store in output
            sample(&destr[n*nrf + p], &desti[n*nrf + p], omr, omi, -phi[p]);
            
            // relaxation until TR 
            relax(omr, omi, t1[n], t2[n], tr[p], p+1);

            // gradient spoiling 
            spoil(omr, omi, omr_copy, omi_copy, p+2);

        } // end pulse loop 

    }

}