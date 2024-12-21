#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
// #define double long double

// #define float double


namespace CDP{
    double get_CDP_w(char x, char y){
        if(x == 'A' and y == 'T')
            return 2;
        if(x == 'T' and y == 'A')
            return 2;
        if(x == 'G' and y == 'C')
            return 3;
        if(x == 'C' and y == 'G')
            return 3;
        if(x == 'G' and y == 'T')
            return 0.8;
        if(x == 'T' and y == 'G')
            return 0.8;
        return 0;
    }

    vector<vector<double> > get_CDP(string seq){
        vector<vector<double>> dp(seq.length(), vector<double>(seq.length(), 0));
        for(int idx = 0; idx < seq.size(); idx++){
            for(int idy = 0; idy < seq.size(); idy++){
                char x = seq[idx];
                char y = seq[idy];
                dp[idx][idy] = get_CDP_w(x, y);
                if(get_CDP_w(x, y) != 0){
                    double TA = 0, TB = 0, a = 1, b = 1;
                    while(idx - a >= 0 && idx - a < seq.size() && idy + a >= 0 && idy + a < seq.size() && get_CDP_w(seq[idx - a], seq[idy + a]) != 0){
                        TA = TA + get_CDP_w(seq[idx-a], seq[idy+a]) * exp(-a * a / 2);
                        a = a + 1;
                    }
                    while(idx + b >= 0 && idx + b < seq.size() && idy - b >= 0 && idy - b < seq.size() && get_CDP_w(seq[idx + b], seq[idy - b]) != 0){
                        TB = TB + get_CDP_w(seq[idx+b], seq[idy-b]) * exp(-b * b / 2 );
                        b = b + 1;
                    }
                    dp[idx][idy] = dp[idx][idy] + TA + TB;
                }
            }
        }
        return dp;
    }
}

PYBIND11_MODULE(CDP, m) {
    m.def("get_CDP", &CDP::get_CDP, "cal single seq CDP matrix");
}