#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define double long double

// #define float double

namespace Sptial_Dis{
    int get_dis(vector<int> x, vector<int> y){
        int sum = 0;
        for(int idx = 0; idx < x.size(); idx++){
            sum = sum + abs(x[idx] - y[idx]);
        }
        return sum;
    }

    vector<int> decimalToBinary(int x){
        vector<int> result;
        while(x){
            result.push_back(x % 2);
            x /= 2;
        }
        return result;
    }

    int bitcount(int n){
        int count = 0;
        while (n) {
            count++;
            n &= (n - 1);
        }
        return count;
    }


    int one_hot_dis(char x1, char x2){
        if(x1 == x2) return 0;
        if(x1 == 'N' || x2 == 'N') return 1;
        return 2;
    }

    vector<int> get_one_hot_distance(char c){
        if(c == 'A'){
            return vector<int> {1, 0, 0, 0};
        }
        if(c == 'C'){
            return vector<int> {0, 1, 0, 0};
        }
        if(c == 'T'){
            return vector<int> {0, 0, 1, 0};
        }
        if(c == 'G'){
            return vector<int> {0, 0, 0, 1};
        }
        return vector<int> {0, 0, 0, 0};
    }

    vector<vector<double>> get_Sptial_Dis(string seq){
        vector<vector<double>> result(seq.length(), vector<double>(seq.length(), 0));
        int seq_length = seq.length();
        for(int i = 0; i < seq_length; i++){
            for(int j = 0; j < seq_length; j++){
                // vector<int> A = decimalToBinary(i);
                // vector<int> B = decimalToBinary(j);
                // while(A.size() < 10) A.insert(A.begin(), 0);
                // while(B.size() < A.size()) B.insert(B.begin(), 0);
                // vector<int> A_O = get_one_hot_distance(seq[i]);
                // vector<int> B_O = get_one_hot_distance(seq[j]);
                int dis1 = (i & j);
                int dis = bitcount(dis1) + one_hot_dis(seq[i], seq[j]);
                // int dis = get_dis(A, B) + get_dis(A_O, B_O);
                result[i][j] = sqrt(dis * 1.0);
            }
        }
        return result;
    }
}

PYBIND11_MODULE(Sptial_Dis, m) {
    m.def("get_Sptial_Dis", &Sptial_Dis::get_Sptial_Dis, "cal single seq DIS matrix");
}