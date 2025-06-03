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

std::string interNeuronToJson(const interNeuron& n) {
    std::ostringstream oss;
    oss << std::setprecision(6) << std::fixed;
    oss << "{"
        << "\"V\":" << n.V << ","
        << "\"m\":" << n.m << ","
        << "\"h\":" << n.h << ","
        << "\"n\":" << n.n << ","
        << "\"w\":" << n.w
        << "}";
    return oss.str();
}

void interToJson(int n, interNeuron* neurons, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    file << "[";
    for (size_t i = 0; i < n; ++i) {
        file << interNeuronToJson(neurons[i]);
        if (i != n - 1) file << ",";
    }
    file << "]";

    file.close();
    std::cout << "Saved to " << filename << std::endl;
}

void saveVectorToJson(const std::vector<std::vector<std::vector<float>>>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    outfile << "[\n"; // Start outer array

    for (size_t i = 0; i < data.size(); ++i) {
        const auto& outer = data[i];
        outfile << "  [\n"; // Start middle array

        for (size_t j = 0; j < outer.size(); ++j) {
            const auto& middle = outer[j];
            outfile << "    [ "; // Start inner array

            for (size_t k = 0; k < middle.size(); ++k) {
                outfile << std::fixed << std::setprecision(4) << middle[k];
                if (k < middle.size() - 1) {
                    outfile << ", ";
                }
            }

            outfile << " ]"; // Close inner array
            if (j < outer.size() - 1) {
                outfile << ",";
            }
            outfile << "\n";
        }

        outfile << "  ]"; // Close middle array
        if (i < data.size() - 1) {
            outfile << ",";
        }
        outfile << "\n";
    }

    outfile << "]"; // Close outer array
    outfile.close();
}
