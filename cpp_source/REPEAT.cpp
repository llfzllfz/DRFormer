#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define double long double

// #define float double

namespace REPEAT{
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

    vector<vector<vector<double>>> get_REPEAT(string seq){
        vector<vector<vector<int>>> dp(5, vector<vector<int>>(seq.length(), vector<int>(seq.length(), 0)));
        vector<vector<int>> MLDP = get_MLDP(seq);
        vector<int> repeat(seq.length(), 0);
        for(int i = 0; i < MLDP.size(); i++){
            int tmp = 0;
            for(int j = 0; j < MLDP[i].size(); j++){
                tmp += (MLDP[i][j] > 0 ? 1 : 0);
            }
            repeat[i] = tmp;
        }
        char lists[] = {'A', 'C', 'T', 'G'};
        int count = 0;
        for(int i = 0; i < 4; i++){
            dp[count] = tree(seq, lists[i], repeat);
            count ++;
        }
        for(int i = 0; i < seq.length(); i++){
            for(int j = 0; j < seq.length(); j++){
                dp[count][i][j] = dp[0][i][j] + dp[1][i][j] + dp[2][i][j] + dp[3][i][j];
            }
        }
        vector<vector<vector<double>>> result(5, vector<vector<double>>(seq.length(), vector<double>(seq.length(), 0)));
        int seq_length = seq.length();
        for(int i = 0; i < seq_length; i++){
            for(int j = 0; j < seq_length; j++){
                result[0][i][j] = dp[0][i][j] * 1.0 / (seq_length * seq_length / 16.0);
                result[1][i][j] = dp[1][i][j] * 1.0 / (seq_length * seq_length / 16.0);
                result[2][i][j] = dp[2][i][j] * 1.0 / (seq_length * seq_length / 8.0);
                result[3][i][j] = dp[3][i][j] * 1.0 / (seq_length * seq_length / 8.0);
                result[4][i][j] = dp[4][i][j] * 1.0 / (seq_length * seq_length * 3.0 / 8.0);
            }
        }
        return result;
    }
}

PYBIND11_MODULE(REPEAT, m) {
    m.def("get_REPEAT", &REPEAT::get_REPEAT, "cal single seq REPEAT matrix");
}
