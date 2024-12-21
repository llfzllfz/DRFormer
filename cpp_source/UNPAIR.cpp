#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define double long double

#define float double

namespace UNPAIR{
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

    /*
    1. 计算区间 i - j 之间没有配对的碱基的数量
    2. 计算区间 i - j 之间没有配对的碱基 A 的数量
    3. ... C ...
    4. ... T ...
    5. ... G ...

    1. 计算区间 i - j 之间配对碱基的重复引用次数
    2. 计算区间 i - j 之间配对剪辑 A 的重复引用次数
    3. ... C ...
    4. ... T ...
    5. ... G ... 
    */

    vector<vector<int>> tree(string seq, char x, vector<int> v){
        int seq_length = seq.length();
        vector<int> dp(seq_length, 0);
        vector<vector<int>> result(seq_length, vector<int>(seq_length, 0));
        for(int i = 0; i < v.size(); i++){
            if(seq[i] == x) dp[i] = v[i];
        }
        vector<int> tr(seq_length + 1, 0);
        for(int i = 1; i < v.size(); i++){
            dp[i] += dp[i-1];
        }
        for(int i = 0; i < seq_length; i++){
            for(int j = i; j < seq_length; j++){
                if(i == 0)
                    result[i][j] = dp[j];
                else
                    result[i][j] = dp[j] - dp[i-1];
                result[j][i] = result[i][j];
            }
        }
        return result;
    }

    // vector<vector<vector<double>>> get_UNPAIR(string seq){
    //     vector<vector<vector<int>>> dp(5, vector<vector<int>>(seq.length(), vector<int>(seq.length(), 0)));
    //     vector<vector<int>> MLDP = get_MLDP(seq);
    //     vector<int> unpair(seq.length(), 0);
    //     for(int i = 0; i < MLDP.size(); i++){
    //         int tmp = 0;
    //         for(int j = 0; j < MLDP[i].size(); j++){
    //             tmp += (MLDP[i][j] > 0 ? 1 : 0);
    //         }
    //         unpair[i] = (tmp > 0 ? 0 : 1);
    //     }
    //     char lists[] = {'A', 'C', 'T', 'G'};
    //     int count = 0;
    //     for(int i = 0; i < 4; i++){
    //         dp[count] = tree(seq, lists[i], unpair);
    //         count ++;
    //     }
    //     for(int i = 0; i < seq.length(); i++){
    //         for(int j = 0; j < seq.length(); j++){
    //             dp[count][i][j] = dp[0][i][j] + dp[1][i][j] + dp[2][i][j] + dp[3][i][j];
    //         }
    //     }
    //     vector<vector<vector<double>>> result(5, vector<vector<double>>(seq.length(), vector<double>(seq.length(), 0)));
    //     int seq_length = seq.length();
    //     for(int i = 0; i < seq_length; i++){
    //         for(int j = 0; j < seq_length; j++){
    //             result[0][i][j] = dp[0][i][j] * 1.0 / (seq_length / 4.0);
    //             result[1][i][j] = dp[1][i][j] * 1.0 / (seq_length / 4.0);
    //             result[2][i][j] = dp[2][i][j] * 1.0 / (seq_length / 4.0);
    //             result[3][i][j] = dp[3][i][j] * 1.0 / (seq_length / 4.0);
    //             result[4][i][j] = dp[4][i][j] * 1.0 / (seq_length);
    //         }
    //     }
    //     return result;
    // }

    int get_count(int i, int j, vector<int> v){
        if(i == 0) return v[i];
        else return v[j] - v[i-1];
    }

    vector<vector<vector<double>>> get_UNPAIR(string seq){
        vector<vector<vector<int>>> dp(5, vector<vector<int>>(seq.length(), vector<int>(seq.length(), 0)));
        vector<vector<int>> MLDP = get_MLDP(seq);
        vector<int> unpair_A(seq.length(), 0);
        vector<int> unpair_C(seq.length(), 0);
        vector<int> unpair_T(seq.length(), 0);
        vector<int> unpair_G(seq.length(), 0);
        for(int i = 0; i < seq.length(); i++){
            if(seq[i] == 'A') unpair_A[i] = 1;
            else if(seq[i] == 'C') unpair_C[i] = 1;
            else if(seq[i] == 'T') unpair_T[i] = 1;
            else if(seq[i] == 'G') unpair_G[i] = 1;
        }
        for(int i = 1; i < seq.length(); i++){
            unpair_A[i] += unpair_A[i-1];
            unpair_C[i] += unpair_C[i-1];
            unpair_T[i] += unpair_T[i-1];
            unpair_G[i] += unpair_G[i-1];
        }
        
        for(int i = 0; i < seq.length(); i++){
            for(int j = i; j < seq.length(); j++){
                // dp[0][i][j] = (get_count(i, j, unpair_T) * 1.0) * 3.0 / ((j - i + 1) * 1.0);
                // dp[1][i][j] = (get_count(i, j, unpair_T) + get_count(i, j, unpair_G) * 1.0) * 3.0 / ((j - i + 1) * 1.0) / 2.0;
                // dp[2][i][j] = (get_count(i, j, unpair_A) + get_count(i, j, unpair_C) * 1.0) * 3.0 / ((j - i + 1) * 1.0) / 2.0;
                // dp[3][i][j] = (get_count(i, j, unpair_C) * 1.0) * 3.0 / ((j - i + 1) * 1.0);
                dp[0][i][j] = (get_count(i, j, unpair_T));
                dp[1][i][j] = (get_count(i, j, unpair_T) + get_count(i, j, unpair_G));
                dp[2][i][j] = (get_count(i, j, unpair_A) + get_count(i, j, unpair_C));
                dp[3][i][j] = (get_count(i, j, unpair_C));

                dp[0][j][i] = dp[0][i][j];
                dp[1][j][i] = dp[1][i][j];
                dp[2][j][i] = dp[2][i][j];
                dp[3][j][i] = dp[3][i][j];

            }
        }
        // return dp;
        for(int i = 0; i < seq.length(); i++){
            for(int j = 0; j < seq.length(); j++){
                dp[4][i][j] = dp[0][i][j] + dp[1][i][j] + dp[2][i][j] + dp[3][i][j];
            }
        }
        vector<vector<vector<double>>> result(5, vector<vector<double>>(seq.length(), vector<double>(seq.length(), 0)));
        int seq_length = seq.length();
        for(int i = 0; i < seq_length; i++){
            for(int j = 0; j < seq_length; j++){
                result[0][i][j] = (dp[0][i][j] * 1.0) / ((seq_length * 1.0) / 1.0);
                result[1][i][j] = (dp[1][i][j] * 1.0) / ((seq_length * 1.0) / 1.0);
                result[2][i][j] = (dp[2][i][j] * 1.0) / ((seq_length * 1.0) / 1.0);
                result[3][i][j] = (dp[3][i][j] * 1.0) / ((seq_length * 1.0) / 1.0);
                result[4][i][j] = (dp[4][i][j] * 1.0) / ((seq_length * 1.0));
            }
        }
        return result;
    }
}

PYBIND11_MODULE(UNPAIR, m) {
    m.def("get_UNPAIR", &UNPAIR::get_UNPAIR, "cal single seq UNPAIR matrix");
}
