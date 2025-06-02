
struct pyramNeuron
{
    float V_s, V_p, V_d, m_s, h_s, n_s, w_s, Ca_s, c1_s, r_s, a_p, b_p, Ca_p, c1_p, r_p, a_d, b_d;
};

struct pyramReceptor{
    float g_I, g_E, g_d;
    bool fire;
};

namespace pyram{

__device__
inline float m_inf(float V){
  return 1.0 / (1.0 + exp(-(V + 22.6) / 9.55));
}
__device__
inline float m_tau(float V){
  return 0.2 * (2.03 - 1.91 / (1 + exp(-(V + 24.1) / 16.7)) - 1.98 / (1 + exp((V + 35.6) / 13.2)));
}
__device__
inline float h_inf(float V){
  return 1.0 / (1.0 + exp((V + 34.2) / 7.07));
}
__device__
inline float h_tau(float V){
  return 43.3 - 42.3 / (1 + exp(-(V + 33.2) / 9.61)) - 42.1 / (1 + exp((V + 40.2) / 10.6));
}
__device__
inline float n_inf(float V){
  return 1.0 / (1.0 + exp(-(V + 23.0) / 17.5));
}
__device__
inline float n_tau(float V){
  return 0.2 * (5.48 - 19.4 / (1.0 + exp(-(V + 51.4) / 23.8))) + 14.9 / (1.0 + exp(-(V + 62.7) / 15.6));
}
__device__
inline float w_inf(float V){
  return 1.0 / (1.0 + exp(-V / 5.0));
}
__device__
inline float w_tau(float V){
  return 1.0;
}
__device__
inline float a_inf(float V){
  return 1.0 / (1.0 + exp(-(V + 5.0) / 10.0));
}
__device__
inline float a_tau(float V){
  return 0.2;
}
__device__
inline float b_inf(float V){
  return 1.0 / (1.0 + exp((V + 58.0) / 8.2));
}
__device__
inline float b_tau(float V){
  return 5.0 + max(0.0, 0.26 * (V + 20.0));
}
__device__
inline float r_inf(float V){
  return 1.0 / (1.0 + exp(-(V + 5.0) / 10.0));
}
__device__
inline float r_tau(float V){
  return 1.0;
}
__device__
inline float Vh(float Ca){
  return 72.0 - 30.0 * log(max(0.1, Ca));
}
__device__
inline float c1_inf(float V, float Ca){
  return 1.0 / (1.0 + exp((Vh(Ca) - V) / 13.0));
}

const float c1_tau = 2.0;
const float Ca_tau = 20.0;

const float vL = -65.0;
const float vNa = 55.0;
const float vK = -90.0;
const float vCa = 120.0;
const float EI = -75.0;

const float Comp_S = 100;
const float Comp_D = 300;
const float A_s = 0.02;
const float A_p = 0.02;
const float A_d = 0.06;

const float I_ext_s = 0.0;
const float I_ext_p = 0.0;
const float I_ext_d = 0.0;

const float exc_tau = 2.0;
const float inh_tau = 2.0;

const float noise_freq = 1;
const float noise_mag = 0.2;
const float noise_tau = 0.5;

__device__
uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__global__
void pyramStep(int n, int id, float h, pyramNeuron* vpn, pyramReceptor* vpr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    pyramNeuron* pn = &vpn[i];
    pyramReceptor* pr = &vpr[i];

    float rval = (wang_hash(i)^wang_hash(id));
    rval /= ((uint)-1);
    rval /= noise_freq*h;
    if(rval<1){
      pr->g_d += noise_mag*(2*rval-1);
    }

    float I_L_s = 0.1; // * (vL - V_s)
    float I_Na_s = 30.0 * pn->m_s * pn->m_s * pn->m_s * pn->h_s; // * (vNa - V_s)
    float I_K_s = 5.0 * pn->n_s * pn->n_s * pn->n_s * pn->n_s; // * (vK - V_s)
    float I_KHT_s = 8.0 * pn->w_s; // * (vK - V_s)
    float I_CaBK_s = 200.0 * pn->c1_s; // * (vK - V_s)
    float I_Ca_s = 0.5 * pn->r_s * pn->r_s; // * (vCa - V_s)

    //Proximal
    float I_L_p = 0.1; // * (vL - V_p)
    float I_A_p = 10.0 * pn->a_p * pn->a_p * pn->a_p * pn->a_p * pn->b_p; // * (vK - V_p)
    float I_CaBK_p = 45.0 * pn->c1_p; // * (vK - V_p)
    float I_Ca_p = 45.0 * pn->r_p * pn->r_p; // * (vCa - V_p)
    float I_exc_p = pr->g_E;
    float I_inh_p = pr->g_I;

    //Distal
    float I_L_d = 0.1; // * (vL - V_d)
    float I_A_d = 10.0 * pn->a_d * pn->a_d * pn->a_d * pn->a_d * pn->b_d; // * (vK - V_d)
    float I_exc_d = pr->g_d;

    float I_s_sum = I_L_s + I_Na_s + I_K_s + I_KHT_s + I_CaBK_s + I_Ca_s;
    float V_s_eq = (I_L_s * vL + I_Na_s * vNa + I_K_s * vK + I_KHT_s * vK + I_CaBK_s * vK + I_Ca_s * vCa);

    float I_p_sum = I_L_p + I_A_p + I_CaBK_p + I_Ca_p + I_exc_p + I_inh_p;
    float V_p_eq = (I_L_p * vL + I_A_p * vK + I_CaBK_p * vK + I_Ca_p * vCa + I_inh_p * EI);

    float I_d_sum = I_L_d + I_A_d + I_exc_d;
    float V_d_eq = (I_L_d * vL + I_A_d * vK);

    // let I_comp_s = (V_p - V_s) / 100
    // let I_comp_d = (V_p - V_d) / 300
    // let I_comp_p = I_comp_s - I_comp_d

    float M_s = I_s_sum + 1 / Comp_S / A_s;
    float M_p = I_p_sum - 1 / Comp_S / A_p + 1 / Comp_D / A_p;
    float M_d = I_d_sum + 1 / Comp_D / A_d;
    V_s_eq += I_ext_s / A_s + pn->V_p / Comp_S / A_s;
    V_p_eq += I_ext_p / A_p + pn->V_d / Comp_D / A_p - pn->V_s / Comp_S / A_p;
    V_d_eq += I_ext_d / A_d + pn->V_p / Comp_D / A_d;
    float V_s_1 = pn->V_s + h * (V_s_eq/M_s - pn->V_s) / (1 / M_s + h);
    float V_p_1 = pn->V_p + h * (V_p_eq/M_p - pn->V_p) / (1 / M_p + h);
    float V_d_1 = pn->V_d + h * (V_d_eq/M_d - pn->V_d) / (1 / M_d + h);

    //Soma
    pn->m_s += h * (m_inf(pn->V_s) - pn->m_s) / (m_tau(pn->V_s) + h);
    pn->h_s += h * (h_inf(pn->V_s) - pn->h_s) / (h_tau(pn->V_s) + h);
    pn->n_s += h * (n_inf(pn->V_s) - pn->n_s) / (n_tau(pn->V_s) + h);
    pn->w_s += h * (w_inf(pn->V_s) - pn->w_s) / (w_tau(pn->V_s) + h);
    pn->c1_s += h * (c1_inf(pn->V_s, pn->Ca_s) - pn->c1_s) / (c1_tau + h);
    pn->Ca_s += h * (0.0002 * I_Ca_s * (vCa - pn->V_s) * Ca_tau - pn->Ca_s) / (Ca_tau + h);
    pn->r_s += h * (r_inf(pn->V_s) - pn->r_s) / (r_tau(pn->V_s) + h);

    //Proximal
    pn->a_p += h * (a_inf(pn->V_p) - pn->a_p) / (a_tau(pn->V_p) + h);
    pn->b_p += h * (b_inf(pn->V_p) - pn->b_p) / (b_tau(pn->V_p) + h);
    pn->c1_p += h * (c1_inf(pn->V_p, pn->Ca_p) - pn->c1_p) / (c1_tau + h);
    pn->Ca_p += h * (0.0002 * I_Ca_p * (vCa - pn->V_p) * Ca_tau - pn->Ca_p) / (Ca_tau + h);
    pn->r_p += h * (r_inf(pn->V_p) - pn->r_p) / (r_tau(pn->V_p) + h);

    //Distal
    pn->a_d += h * (a_inf(pn->V_d) - pn->a_d) / (a_tau(pn->V_d) + h);
    pn->b_d += h * (b_inf(pn->V_d) - pn->b_d) / (b_tau(pn->V_d) + h);

    pr->g_E += h * (-pr->g_E) / (exc_tau + h);
    pr->g_I += h * (-pr->g_I) / (inh_tau + h);
    pr->fire = pn->V_s<=0.0 && V_s_1>0.0;
    pr->g_d += h * (-pr->g_d) / (noise_tau + h);

    pn->V_s = V_s_1;
    pn->V_p = V_p_1;
    pn->V_d = V_d_1;
}

void initPyramNeuron(pyramNeuron* pn) {
    pn->V_s = -64.6240844727f;
    pn->V_p = -64.5471725464f;
    pn->V_d = -64.8385238647f;
    pn->m_s = 0.0121233119f;
    pn->h_s = 0.9866571426f;
    pn->n_s = 0.0848252103f;
    pn->w_s = 2.4368e-06f;
    pn->Ca_s = 2.4334e-06f;
    pn->c1_s = 1.343e-07f;
    pn->r_s = 0.0025670978f;
    pn->a_p = 0.0025868677f;
    pn->b_p = 0.6896412373f;
    pn->Ca_p = 0.0002222956f;
    pn->c1_p = 1.351e-07f;
    pn->r_p = 0.0025868667f;
    pn->a_d = 0.0025127728f;
    pn->b_d = 0.6971943974f;
}

void initPyramReceptor(pyramReceptor* pr){
  pr[0] = {};
}

}