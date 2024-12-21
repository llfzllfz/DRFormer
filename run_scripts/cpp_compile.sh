g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/MLDP.cpp -o bin/MLDP`python3-config --extension-suffix`
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/CDP.cpp -o bin/CDP`python3-config --extension-suffix`
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/DIS.cpp -o bin/DIS`python3-config --extension-suffix`
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/UFOLD.cpp -o bin/UFOLD`python3-config --extension-suffix`
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/UNPAIR.cpp -o bin/UNPAIR`python3-config --extension-suffix`
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/REPEAT.cpp -o bin/REPEAT`python3-config --extension-suffix`
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/Sptial_Dis.cpp -o bin/Sptial_Dis`python3-config --extension-suffix`
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp_source/Entropy.cpp -o bin/Entropy`python3-config --extension-suffix`
