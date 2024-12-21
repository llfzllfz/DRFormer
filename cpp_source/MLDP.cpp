#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define float double

namespace MLDP{
    int check_MLDP(char x, char y){
        if(x == 'A' && y == 'T')
            return 1;
        if(x == 'T' && y == 'A')
            return 1;
        if(x == 'C' && y == 'G')
            return 1;
        if(x == 'G' && y == 'C')
            return 1;
        if(x == 'G' && y == 'T')
            return 1;
        if(x == 'T' && y == 'G')
            return 1;
        return 0;
    }

    vector<vector<int>> get_MLDP(string seq){
        vector<vector<int>> dp(seq.length(), vector<int>(seq.length() + 1, 0));
        for(int i = 0; i < seq.length(); i++){
            dp[0][i] = check_MLDP(seq[0], seq[i]);
        }
        for(int idx = 1; idx < seq.length(); idx++){
            for(int idy = seq.length() - 1; idy > idx; idy--){
                if(check_MLDP(seq[idx], seq[idy]) == 1){
                    dp[idx][idy] = dp[idx - 1][idy + 1] + check_MLDP(seq[idx], seq[idy]);
                }
            }
        }
        for(int idx = 0; idx < dp.size(); idx++){
            dp[idx].pop_back();
        }
        for(int idx = dp.size() - 2; idx >= 0; idx--){
            for(int idy = 1; idy < dp[0].size(); idy++){
                if(dp[idx][idy] != 0)
                    dp[idx][idy] = max(dp[idx][idy], dp[idx+1][idy-1]);
            }
        }
        for(int idx = 0; idx < dp.size(); idx++){
            for(int idy = idx + 1; idy < dp[idx].size(); idy++){
                dp[idy][idx] = dp[idx][idy];
            }
        }
        return dp;
    }
}

PYBIND11_MODULE(MLDP, m) {
    m.def("get_MLDP", &MLDP::get_MLDP, "cal single seq MLDP matrix");
}
