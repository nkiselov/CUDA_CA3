#include <iostream>
#include "pyramNeuron.cu"
#include "interNeuron.cu"
#include "io.cpp"

int main(void)
{
    const int N = 1<<10;
    const int BLOCK = 256;
    const int ITERS = 40000;

    pyramNeuron *vpn, *d_vpn;
    vpn = (pyramNeuron*)malloc(N*sizeof(pyramNeuron));
    cudaMalloc(&d_vpn, N*sizeof(pyramNeuron));

    pyramReceptor *vpr, *d_vpr;
    vpr = (pyramReceptor*)malloc(N*sizeof(pyramReceptor));
    cudaMalloc(&d_vpr, N*sizeof(pyramReceptor));

    for (int i = 0; i < N; i++) pyram::initPyramNeuron(&vpn[i]);
    for (int i = 0; i < N; i++) pyram::initPyramReceptor(&vpr[i]);

    pyramNeuron* result;
    result = (pyramNeuron*)malloc(ITERS*sizeof(pyramNeuron));
    for(int i=0; i<ITERS; i++){
      // if(i==200){
      //   for(int i=0; i<N; i++) vpr[i].g_E+=0.275;
      // }

      cudaMemcpy(d_vpr, vpr, N*sizeof(pyramReceptor), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vpn, vpn, N*sizeof(pyramNeuron), cudaMemcpyHostToDevice);
      
      pyram::pyramStep<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(N, i, 0.1f, d_vpn, d_vpr);

      cudaMemcpy(vpn, d_vpn, N*sizeof(pyramNeuron), cudaMemcpyDeviceToHost);
      cudaMemcpy(vpr, d_vpr, N*sizeof(pyramReceptor), cudaMemcpyDeviceToHost);

      memcpy(&result[i],&vpn[0],sizeof(pyramNeuron));
    }

    pyramToJson(ITERS,result,"result.json");

    cudaFree(d_vpn);
    free(vpn);
    cudaFree(d_vpr);
    free(vpr);
    free(result);
}