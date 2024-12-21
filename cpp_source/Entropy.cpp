#include "CDP.cpp"
#include "DIS.cpp"
#include "UFOLD.cpp"
#include "UNPAIR.cpp"
#include "MLDP.cpp"
#include "Sptial_Dis.cpp"
#include "REPEAT.cpp"
#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define double long double
// #define float double

vector<vector<double>> transform(vector<vector<int>> V);
// vector<vector<vector<double>>> GET_ALL_CHANNEL(string seq, int NMLDP = 1, int NDIS = 1, int NCDP = 1, int NUFOLD = 1, int NUNPAIR = 1, int NREPEAT = 1, int NSptial_Dis = 1);
vector<double> GET_ALL_CHANNEL_ENTROPY(string seq);
vector<vector<vector<double>>> GET_ALL_CHANNEL_ENTROPY_VECTOR(string seq);
vector<pair<double, double>> TULPLE(vector<vector<double> > V);
double ENTROPY(vector<pair<double, double> > V);
vector<vector<double>> ENTROPY_vector(vector<pair<double, double> > V);

vector<vector<double>> transform(vector<vector<int>> V){
    vector<vector<double>> result;
    for(int i = 0; i < V.size(); i++){
        vector<double> tmp;
        for(int j = 0; j < V[i].size(); j++){
            tmp.push_back((double)(V[i][j]));
        }
        result.push_back(tmp);
    }
    return result;
}

vector<vector<vector<double>>> GET_ALL_CHANNEL(string seq, int NMLDP = 1, int NDIS = 1, int NCDP = 1, int NSptial_Dis = 1, int NUFOLD = 1, int NUNPAIR = 1, int NREPEAT = 1, int NUFOLD_ADD_UNPAIR = 0){
    vector<vector<vector<double>>> result;
    if(NMLDP == 1){
        vector<vector<int>> MLDP = MLDP::get_MLDP(seq);
        result.push_back(transform(MLDP));
        // for(int i = 0; i < seq.length(); i++){
        //     for(int j = 0; j < seq.length(); j++){
        //         if(result[0][i][j] > 7)
        //             result[0][i][j] = log2(result[0][i][j]);
        //     }
        // }
    }
    if(NDIS == 1){
        vector<vector<int>> DIS = DIS::calculate_distance_matrix(seq);
        result.push_back(transform(DIS));
    }
    if(NCDP == 1){
        vector<vector<double>> CDP = CDP::get_CDP(seq);
        result.push_back(CDP);
    }
    if(NSptial_Dis == 1){
        vector<vector<double>> Sptial_Dis = Sptial_Dis::get_Sptial_Dis(seq);
        result.push_back(Sptial_Dis);
    }
    if(NUFOLD == 1){
        vector<vector<vector<int>>> UFOLD = UFOLD::get_UFold(seq);
        if(NUFOLD_ADD_UNPAIR == 1){
            vector<vector<vector<double>>> UFOLD_ADD_UNPAIR = UNPAIR::get_UNPAIR(seq);
            vector<vector<vector<double>>> UFOLD_double;
            for(int i = 0; i < UFOLD.size(); i++){
                UFOLD_double.push_back(transform(UFOLD[i]));
            }
            // cout << fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);
            // for(int i = 0; i < 4; i++){
            //     for(int j = 0; j < UFOLD_ADD_UNPAIR[i].size(); j++){
            //         for(int k = 0; k < UFOLD_ADD_UNPAIR[i][j].size(); k++){
            //             cout << UFOLD_ADD_UNPAIR[i][j][k] << ' ';
            //         }
            //         cout << endl;
            //     }
            //     cout << endl;
            // }
            for(int i = 0; i < UFOLD_double.size(); i++){
                if(i == 0 || i == 1 || i == 2 || i == 3){ // add A
                    for(int j = 0; j < UFOLD_double[i].size(); j++){
                        for(int k = 0; k < UFOLD_double[i][j].size(); k++){
                            UFOLD_double[i][j][k] += 0.001 * UFOLD_ADD_UNPAIR[0][j][k];
                        }
                    }
                }
                if(i == 1 || i == 4 || i == 5 || i == 6){ // add C
                    for(int j = 0; j < UFOLD_double[i].size(); j++){
                        for(int k = 0; k < UFOLD_double[i][j].size(); k++){
                            UFOLD_double[i][j][k] += 0.001 * UFOLD_ADD_UNPAIR[1][j][k];
                        }
                    }
                }
                if(i == 2 || i == 5 || i == 7 || i == 8){ // add T
                    for(int j = 0; j < UFOLD_double[i].size(); j++){
                        for(int k = 0; k < UFOLD_double[i][j].size(); k++){
                            UFOLD_double[i][j][k] += 0.001 * UFOLD_ADD_UNPAIR[2][j][k];
                        }
                    }
                }
                if(i == 3 || i == 6 || i == 8 || i == 9){ // add G
                    for(int j = 0; j < UFOLD_double[i].size(); j++){
                        for(int k = 0; k < UFOLD_double[i][j].size(); k++){
                            UFOLD_double[i][j][k] += 0.001 * UFOLD_ADD_UNPAIR[3][j][k];
                        }
                    }
                }
            }
            
            // for(int j = 0; j < UFOLD_double[2].size(); j++){
            //     for(int k = 0; k < UFOLD_double[2][j].size(); k++){
            //         cout << UFOLD_double[2][j][k] << ' ' ;
            //     }
            //     cout << endl;
            // }
            for(int i = 0; i < UFOLD_double.size(); i++){
                result.push_back(UFOLD_double[i]);
            }
        }
        else{
            for(int i = 0; i < UFOLD.size(); i++){
                result.push_back(transform(UFOLD[i]));
            }
        }
    }
    if(NUNPAIR == 1){
        vector<vector<vector<double>>> UNPAIR = UNPAIR::get_UNPAIR(seq);
        for(int i = 0; i < UNPAIR.size(); i++){
            result.push_back(UNPAIR[i]);
        }
    }
    if(NREPEAT == 1){
        vector<vector<vector<double>>> REPEAT = REPEAT::get_REPEAT(seq);
        for(int i = 0; i < REPEAT.size(); i++){
            result.push_back(REPEAT[i]);
        }
    }
    return result;
}

vector<double> GET_ALL_CHANNEL_ENTROPY(string seq){
    vector<vector<vector<double>>> result = GET_ALL_CHANNEL(seq);
    vector<double> entropy;
    for(int i = 0; i < result.size(); i++){
        vector<pair<double, double>> V_entropy = TULPLE(result[i]);
        entropy.push_back(ENTROPY(V_entropy));
    }
    return entropy;
}

vector<vector<vector<double>>> GET_ALL_CHANNEL_ENTROPY_VECTOR(string seq){
    vector<vector<vector<double>>> result = GET_ALL_CHANNEL(seq);
    vector<vector<vector<double>>> entropy;
    for(int i = 0; i < result.size(); i++){
        vector<pair<double, double>> V_entropy = TULPLE(result[i]);
        vector<vector<double>> tmp = ENTROPY_vector(V_entropy);
        entropy.push_back(tmp);
    }
    return entropy;
}

vector<pair<double, double>> TULPLE(vector<vector<double> > V){
    vector<pair<double, double> > result;
    for(int i = 1; i < V.size() - 2; i++){
        for(int j = 1; j < V[i].size() - 2; j++){
            double sum_1 = 0;
            double sum_2 = 0;
            for(int k = -1; k <= 2; k++){
                sum_2 += V[i-1][j+k];
                sum_2 += V[i+2][j+k];
            }
            for(int k = 0; k <= 1; k++){
                sum_1 += V[i][j+k];
                sum_1 += V[i+1][j+k];
                sum_2 += V[i+k][j-1];
                sum_2 += V[i+k][j+2];
            }
            pair<double, double> tmp(sum_1, sum_2);
            result.push_back(tmp);
        }
    }
    return result;
}

double ENTROPY(vector<pair<double, double> > V){
    map<pair<double, double>, int > mp;
    for(int i = 0; i < V.size(); i++){
        mp[V[i]]++;
    }
    double entropy = 0;
    for(int i = 0; i < V.size(); i++){
        if(mp[V[i]] == 0) continue;
        double P = mp[V[i]] * 1.0 / V.size();
        double entropy_tmp = -P * log2(P);
        entropy += entropy_tmp;
        mp[V[i]] = 0;
    }
    return entropy;
}

vector<vector<double>> ENTROPY_vector(vector<pair<double, double> > V){
    int length = int(sqrt(V.size()));
    vector<vector<double>> result(length, vector<double>(length, 0));
    map<pair<double, double>, int > mp;
    for(int i = 0; i < V.size(); i++){
        mp[V[i]]++;
    }
    for(int i = 0; i < length; i++){
        for(int j = 0; j < length; j++){
            double P = mp[V[i * length + j]] * 1.0 / V.size();
            double entropy_tmp = -P * log2(P);
            result[i][j] = entropy_tmp;
        }
    }
    return result;
}

PYBIND11_MODULE(Entropy, m) {
    m.def("GET_ALL_CHANNEL", &GET_ALL_CHANNEL, "cal GET_ALL_CHANNEL");
    m.def("GET_ALL_CHANNEL_ENTROPY", &GET_ALL_CHANNEL_ENTROPY, "cal GET_ALL_CHANNEL_ENTROPY");
    m.def("GET_ALL_CHANNEL_ENTROPY_VECTOR", &GET_ALL_CHANNEL_ENTROPY_VECTOR, "cal GET_ALL_CHANNEL_ENTROPY_VECTOR");
}