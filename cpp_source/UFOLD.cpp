#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define float double



namespace UFOLD{
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
        return vector<double> {0, 0, 0, 0};
    }

    vector<vector<int>> get_UFold_single(string seq, char x1, char x2){
        vector<vector<int>> dp(seq.length(), vector<int>(seq.length(), 0));
        for(int idx = 0; idx < seq.length(); idx++){
            for(int idy = 0; idy < seq.length(); idy++){
                if((seq[idx] == x1 && seq[idy] == x2) || (seq[idx] == x2 && seq[idy] == x1))
                    dp[idx][idy] = 1;
            }
        }
        return dp;
    }

    vector<vector<vector<int>>> get_UFold(string seq){
        vector<vector<vector<int>>> dp(10, vector<vector<int>>(seq.length(), vector<int>(seq.length(), 0)));
        char lists[] = {'A', 'C', 'T', 'G'};
        int count = 0;
        for(int i = 0; i < 4; i++){
            for(int j = i; j < 4; j++){
                dp[count] = get_UFold_single(seq, lists[i], lists[j]);
                count ++;
            }
        }
        return dp;
    }
}

PYBIND11_MODULE(UFOLD, m) {
    m.def("get_UFold", &UFOLD::get_UFold, "cal single seq UFold matrix");
}
