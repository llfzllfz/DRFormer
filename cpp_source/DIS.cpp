#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define double long double

// #define float double


namespace DIS{
    int get_mhd(vector<double> x, vector<double> y){
        int sum = 0;
        for(int idx = 0; idx < x.size(); idx++){
            // sum = sum + abs(x[idx] - y[idx]);
            if(x[idx] != y[idx])
                sum = sum + 1;
        }
        sum = sum;
        return sum;
    }

    vector<double> get_one_hot_distance(char c){
        if(c == 'A'){
            return vector<double> {1, 0, 0, 0};
        }
        if(c == 'C'){
            return vector<double> {0, 1, 0, 0};
        }
        if(c == 'T'){
            return vector<double> {0, 0, 1, 0};
        }
        if(c == 'G'){
            return vector<double> {0, 0, 0, 1};
        }
        return vector<double> {0.25, 0.25, 0.25, 0.25};
    }

    unordered_map <int, int> get_mhd_dis(){
        unordered_map<int, int> mp;
        vector<char> lists = {'A', 'C', 'T', 'G', 'N'};
        for(auto& x : lists){
            for(auto& y : lists){
                mp[x * y] = get_mhd(get_one_hot_distance(x), get_one_hot_distance(y));
                mp[x * y] = get_mhd(get_one_hot_distance(x), get_one_hot_distance(y));
            }
        }
        return mp;
    }

    vector<vector<int>> calculate_distance_matrix(string seq){
        unordered_map <int, int> mp = get_mhd_dis();
        vector<vector<int>> dp(seq.length(), vector<int>(seq.length(), 0));
        for(int idx = 0; idx < seq.size(); idx++){
            for(int idy = 0; idy < seq.size(); idy++){
                dp[idx][idy] = (mp[seq[idx] * seq[idy]]);
            }
        }
        return dp;
    }
}

PYBIND11_MODULE(DIS, m) {
    m.def("calculate_distance_matrix", &DIS::calculate_distance_matrix, "cal single seq DIS matrix");
}