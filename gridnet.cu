int calc_delay(float d);

struct synapse{
    int from, delay;
    float weight;
    long long input;
};


__global__
void synapseStepPy2In(int n, synapse* vsyn, int* val, pyramReceptor* vpr, interReceptor* vir)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    int st = i>0?val[i-1]:0;
    int en = val[i];
    for(int j=st; j<en; j++){
        if(vpr[vsyn[j].from].fire){
            vsyn[j].input|=(1<<vsyn[j].delay);
        }
        vsyn[j].input>>=1;
        if(vsyn[j].input&1){
            vir[i].g_E+=vsyn[j].weight;
        }
    }
}

__global__
void synapseStepPy2PyS0(int n, synapse* vsyn, int* val, pyramReceptor* vpr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    int st = i>0?val[i-1]:0;
    int en = val[i];
    for(int j=st; j<en; j++){
        if(vpr[vsyn[j].from].fire){
            vsyn[j].input|=(1ll<<vsyn[j].delay);
        }
    }
}

__global__
void synapseStepPy2PyS1(int n, synapse* vsyn, int* val, pyramReceptor* vpr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    int st = i>0?val[i-1]:0;
    int en = val[i];
    for(int j=st; j<en; j++){
        vsyn[j].input>>=1;
        if(vsyn[j].input&1){
            vpr[i].g_E+=vsyn[j].weight;
        }
    }
}

__global__
void synapseStepIn2Py(int n, synapse* vsyn, int* val, pyramReceptor* vpr, interReceptor* vir)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    int st = i>0?val[i-1]:0;
    int en = val[i];
    for(int j=st; j<en; j++){
        if(vir[vsyn[j].from].fire){
            vsyn[j].input|=(1<<vsyn[j].delay);
        }
        vsyn[j].input>>=1;
        if(vsyn[j].input&1){
            vpr[i].g_I+=vsyn[j].weight;
        }
    }
}

#define RANDNUM ((float) rand() / (RAND_MAX))

class gridnet{
    const int BLOCK = 256;

    int size,pyN,inN;
    float inRad,dt;
    int stepInd;

    synapse *py2in, *in2py, *py2py;
    synapse *d_py2in, *d_in2py, *d_py2py;
    int *a_py2in, *a_in2py, *a_py2py;
    int *d_a_py2in, *d_a_in2py, *d_a_py2py;

    pyramNeuron *vpn, *d_vpn;
    pyramReceptor *vpr, *d_vpr;
    interNeuron *vin, *d_vin;
    interReceptor *vir, *d_vir;

    float dist(std::pair<float,float>& pa, std::pair<float,float>& pb){
        return sqrt((pa.first-pb.first)*(pa.first-pb.first) + (pa.second-pb.second)*(pa.second-pb.second));
    }

    void linearize(std::vector<std::vector<synapse>>& syn, synapse** h_vec, synapse** d_vec, int** a_vec, int** d_a_vec){
        int sz = 0;
        for(std::vector<synapse>& v:syn) sz+=v.size();
        *h_vec = (synapse*)malloc(sz*sizeof(synapse));
        *a_vec = (int*)malloc(syn.size()*sizeof(int));
        int ind = 0;
        int aind = 0;
        for(std::vector<synapse>& v:syn){
            for(synapse& s:v){
                (*h_vec)[ind] = s;
                ind++;
            }
            (*a_vec)[aind] = ind;
            aind++;
        }
        cudaMalloc(d_vec, sz*sizeof(synapse));
        cudaMalloc(d_a_vec, syn.size()*sizeof(int));
        
        cudaMemcpy(*d_vec, *h_vec, sz*sizeof(synapse), cudaMemcpyHostToDevice);
        cudaMemcpy(*d_a_vec, *a_vec, syn.size()*sizeof(int), cudaMemcpyHostToDevice);
    }
public:

    gridnet(int size, int inN, float inRad, float dt): size(size),pyN(size*size),inN(inN),inRad(inRad),dt(dt),stepInd(0){
        std::vector<std::pair<float,float>> pyPos(pyN), inPos(inN);
        for(int i=0; i<pyN; i++){
            pyPos[i].first = RANDNUM;
            pyPos[i].second = RANDNUM;
        }
        for(int i=0; i<inN; i++){
            inPos[i].first = RANDNUM;
            inPos[i].second = RANDNUM;
        }
        std::vector<std::vector<synapse>> v_py2in(inN), v_py2py(pyN), v_in2py(pyN);
        for(int i=0; i<pyN; i++){
            for(int j=0; j<inN; j++){
                float d = dist(pyPos[i],inPos[j]);
                if(d>=inRad) continue;
                v_in2py[i].push_back({j,calc_delay(d),0.3,0});
                v_py2in[j].push_back({i,calc_delay(d),0.3,0});
            }
        }
        const int dx[4] = {-1,0,1,0};
        const int dy[4] = {0,1,0,-1};
        for(int i=0; i<size; i++){
            for(int j=0; j<size; j++){
                int ind = i*size+j;
                for(int k=0; k<4; k++){
                    int ni = i+dx[k];
                    int nj = j+dy[k];
                    if(ni<0 || ni>=size || nj<0 || nj>=size) continue;
                    int nind = ni*size+nj;
                    float d = dist(pyPos[ind],pyPos[nind]);
                    v_py2py[ind].push_back({nind,calc_delay(d),0.1});
                }
            }
        }
        linearize(v_py2in,&py2in,&d_py2in,&a_py2in,&d_a_py2in);
        linearize(v_in2py,&in2py,&d_in2py,&a_in2py,&d_a_in2py);
        linearize(v_py2py,&py2py,&d_py2py,&a_py2py,&d_a_py2py);

        vpn = (pyramNeuron*)malloc(pyN*sizeof(pyramNeuron));
        cudaMalloc(&d_vpn, pyN*sizeof(pyramNeuron));

        vpr = (pyramReceptor*)malloc(pyN*sizeof(pyramReceptor));
        cudaMalloc(&d_vpr, pyN*sizeof(pyramReceptor));

        for (int i = 0; i < pyN; i++) pyram::initPyramNeuron(&vpn[i]);
        for (int i = 0; i < pyN; i++) pyram::initPyramReceptor(&vpr[i]);

        cudaMemcpy(d_vpr, vpr, pyN*sizeof(pyramReceptor), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vpn, vpn, pyN*sizeof(pyramNeuron), cudaMemcpyHostToDevice);


        vin = (interNeuron*)malloc(inN*sizeof(interNeuron));
        cudaMalloc(&d_vin, inN*sizeof(interNeuron));

        vir = (interReceptor*)malloc(inN*sizeof(interReceptor));
        cudaMalloc(&d_vir, inN*sizeof(interReceptor));

        for (int i = 0; i < inN; i++) inter::initInterNeuron(&vin[i]);
        for (int i = 0; i < inN; i++) inter::initInterReceptor(&vir[i]);

        cudaMemcpy(d_vir, vir, inN*sizeof(interReceptor), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vin, vin, inN*sizeof(interNeuron), cudaMemcpyHostToDevice);
    }

    ~gridnet(){
        free(py2in);
        free(in2py);
        free(py2py);
        cudaFree(d_py2in);
        cudaFree(d_in2py);
        cudaFree(d_py2py);
        free(a_py2in);
        free(a_in2py);
        free(a_py2py);
        cudaFree(d_a_py2in);
        cudaFree(d_a_in2py);
        cudaFree(d_a_py2py);
        cudaFree(d_vin);
        free(vin);
        cudaFree(d_vir);
        free(vir);
        cudaFree(d_vpn);
        free(vpn);
        cudaFree(d_vpr);
        free(vpr);
    }

    void excite(int i, int j, float val){
        cudaMemcpy(vpr, d_vpr, pyN*sizeof(pyramReceptor), cudaMemcpyDeviceToHost);
        vpr[size*i+j].g_E+=val;
        cudaMemcpy(d_vpr, vpr, pyN*sizeof(pyramReceptor), cudaMemcpyHostToDevice);
    }

    void gridStep(){
        pyram::pyramStep<<<(pyN+BLOCK-1)/BLOCK, BLOCK>>>(pyN, stepInd, dt, d_vpn, d_vpr);
        stepInd++;
        inter::interStep<<<(inN+BLOCK-1)/BLOCK, BLOCK>>>(inN, dt, d_vin, d_vir);

        synapseStepPy2In<<<(inN+BLOCK-1)/BLOCK, BLOCK>>>(inN,d_py2in,d_a_py2in,d_vpr,d_vir);
        synapseStepPy2PyS0<<<(pyN+BLOCK-1)/BLOCK, BLOCK>>>(pyN,d_py2py,d_a_py2py,d_vpr);
        synapseStepPy2PyS1<<<(pyN+BLOCK-1)/BLOCK, BLOCK>>>(pyN,d_py2py,d_a_py2py,d_vpr);
        synapseStepIn2Py<<<(pyN+BLOCK-1)/BLOCK, BLOCK>>>(pyN,d_in2py,d_a_in2py,d_vpr,d_vir);

        // cudaMemcpy(py2py, d_py2py, 100*sizeof(synapse), cudaMemcpyDeviceToHost);
        // long long mx = 0;
        // for(int i=0; i<100; i++) mx=max(mx,py2py[i].input);
        // std::cout<<mx<<std::endl;
    }

    std::vector<std::vector<float>> getSomaV(){
        cudaMemcpy(vpn, d_vpn, pyN*sizeof(pyramNeuron), cudaMemcpyDeviceToHost);
        std::vector<std::vector<float>> res(size,std::vector<float>(size));
        for(int i=0; i<size; i++){
            for(int j=0; j<size; j++){
                res[i][j] = vpn[size*i+j].V_s;
            }
        }
        return res;
    }
};
