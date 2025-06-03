#include "headers.h"

int calc_delay(float d){
    return max(1,min(63,(int)(d*127/sqrt(2.0)+RANDNUM*30)));
}

int main(){
    gridnet* gn = new gridnet(20,1,0.1,10);
    std::vector<std::vector<std::vector<float>>> history;
    gn->excite(5,5,1);
    for(int i=0; i<1000; i++){
        for(int j=0; j<10; j++) gn->gridStep();
        history.push_back(gn->getSomaV());
    }
    delete gn;
    saveVectorToJson(history,"grid.json");
}