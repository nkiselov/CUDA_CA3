
struct interNeuron
{
    float V,m,h,n,w;
};

struct interReceptor{
    float g_E;
    bool fire;
};

namespace inter{

__device__
inline float am(float V){
  return (V+22.0)/(1.0-exp(-(V+22.0)/10.0));
}
__device__
inline float bm(float V){
    return 40.0*exp(-(V+47.0)/18.0);
}
__device__
inline float ah(float V){
    return 0.7*exp(-(V+34.0)/20.0);
}
__device__
inline float bh(float V){
    return 10.0/(1.0+exp(-(V+4.0)/10.0));
}
__device__
inline float an(float V){
    return 0.15*(V+15.0)/(1.0-exp(-(V+15.0)/10.0));
}
__device__
inline float bn(float V){
    return 0.2*exp(-(V+25.0)/80.0);
}
__device__
inline float w_inf(float V){
    return 1.0/(1.0+exp(-V/5.0));
}

const float w_tau = 1.0;

const float vL = -65.0;
const float vNa = 55.0;
const float vK = -90.0;

const float I_ext = 0.0;

const float exc_tau = 2.0;

__global__
void interStep(int n, float h, interNeuron* vin, interReceptor* vir)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    interNeuron* in = &vin[i];
    interReceptor* ir = &vir[i];

    float I_L = 0.1; // * (vL - V)
    float I_Na = 30.0 * in->m * in->m * in->m * in->h; // * (vNa - V)
    float I_K = 5.0 * in->n * in->n * in->n * in->n; // * (vK - V)
    float I_KHT = 8.0 * in->w; // * (vK - V)
    float I_exc = ir->g_E;

    float I_sum = I_L + I_Na + I_K + I_KHT + I_exc;
    float V_eq = (I_L * vL + I_Na * vNa + I_K * vK + I_KHT * vK);

    V_eq+=I_ext;
    float V_1 = in->V + h * (V_eq/I_sum-in->V)/(1/I_sum+h);

    float m_mul = am(in->V)+bm(in->V);
    float h_mul = ah(in->V)+bh(in->V);
    float n_mul = an(in->V)+bn(in->V);

    in->m += h * (am(in->V)/m_mul-in->m)/(1/m_mul+h);
    in->h += h * (ah(in->V)/h_mul-in->h)/(1/h_mul+h);
    in->n += h * (an(in->V)/n_mul-in->n)/(1/n_mul+h);
    in->w += h * (w_inf(in->V)-in->w)/(w_tau+h);

    ir->g_E += h * (-ir->g_E) / (exc_tau + h);
    ir->fire = in->V<=0.0 && V_1>0.0;

    in->V = V_1;
}

void initInterNeuron(interNeuron* in) {
    in->V = -65.350777f;
    in->m = 0.005164f; 
    in->h = 0.993603f;
    in->n = 0.129927;
    in->w = 2e-06f;
}

void initInterReceptor(interReceptor* ir){
  ir[0] = {};
}

}