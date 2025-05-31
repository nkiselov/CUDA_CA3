#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

std::string pyramNeuronToJson(const pyramNeuron& n) {
    std::ostringstream oss;
    oss << std::setprecision(10) << std::fixed;
    oss << "{"
        << "\"V_s\":" << n.V_s << ","
        << "\"V_p\":" << n.V_p << ","
        << "\"V_d\":" << n.V_d << ","
        << "\"m_s\":" << n.m_s << ","
        << "\"h_s\":" << n.h_s << ","
        << "\"n_s\":" << n.n_s << ","
        << "\"w_s\":" << n.w_s << ","
        << "\"Ca_s\":" << n.Ca_s << ","
        << "\"c1_s\":" << n.c1_s << ","
        << "\"r_s\":" << n.r_s << ","
        << "\"a_p\":" << n.a_p << ","
        << "\"b_p\":" << n.b_p << ","
        << "\"Ca_p\":" << n.Ca_p << ","
        << "\"c1_p\":" << n.c1_p << ","
        << "\"r_p\":" << n.r_p << ","
        << "\"a_d\":" << n.a_d << ","
        << "\"b_d\":" << n.b_d
        << "}";
    return oss.str();
}

void pyramToJson(int n, pyramNeuron* neurons, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    file << "[";
    for (size_t i = 0; i < n; ++i) {
        file << pyramNeuronToJson(neurons[i]);
        if (i != n - 1) file << ",";
    }
    file << "]";

    file.close();
    std::cout << "Saved to " << filename << std::endl;
}