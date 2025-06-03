#include "headers.h"

int main(void)
{
    const int N = 1<<10;
    const int BLOCK = 256;
    const int ITERS = 10000;

    interNeuron *vin, *d_vin;
    vin = (interNeuron*)malloc(N*sizeof(interNeuron));
    cudaMalloc(&d_vin, N*sizeof(interNeuron));

    interReceptor *vir, *d_vir;
    vir = (interReceptor*)malloc(N*sizeof(interReceptor));
    cudaMalloc(&d_vir, N*sizeof(interReceptor));

    for (int i = 0; i < N; i++) inter::initInterNeuron(&vin[i]);
    for (int i = 0; i < N; i++) inter::initInterReceptor(&vir[i]);

    interNeuron* result;
    result = (interNeuron*)malloc(ITERS*sizeof(interNeuron));
    for(int i=0; i<ITERS; i++){
      if(i==200){
        for(int i=0; i<N; i++) vir[i].g_E+=1;
      }

      cudaMemcpy(d_vir, vir, N*sizeof(interReceptor), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vin, vin, N*sizeof(interNeuron), cudaMemcpyHostToDevice);

      inter::interStep<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(N, 0.1f, d_vin, d_vir);

      cudaMemcpy(vin, d_vin, N*sizeof(interNeuron), cudaMemcpyDeviceToHost);
      cudaMemcpy(vir, d_vir, N*sizeof(interReceptor), cudaMemcpyDeviceToHost);

      memcpy(&result[i],&vin[0],sizeof(interNeuron));
    }

    interToJson(ITERS,result,"result.json");

    cudaFree(d_vin);
    free(vin);
    cudaFree(d_vir);
    free(vir);
    free(result);
}