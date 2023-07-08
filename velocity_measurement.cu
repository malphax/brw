//#define GPU_SUPPORT
#include "cubrw.hpp"
#include "matplotlibcpp.h"
#include <fstream>
#include <ranges>
#include <iterator>
#include <iostream>

namespace plt = matplotlibcpp;

int main()
{
    int saving_period = 1000;
    u_field u(10000, {100, 400, 700, 1000} , saving_period);
    std::cout << "T=";
    unsigned long T = 10000;
    std::cin >> T;
    u.fill_checkpoints(T, 3000);

    int last_checkpoint = (T/saving_period) * saving_period;

    std::vector<double> y(u.cbegin_y(last_checkpoint), u.cend_y(last_checkpoint));

    plt::plot(y);
    plt::show();

    // std::ifstream infile("w.txt");
    // infile >> u;
    // infile.close();

    // std::ofstream file("w2.txt");
    // file << u;
    // file.close();

    /*//measure velocity
    std::vector<double> mt;
    std::vector<double> tvec;
    std::vector<double> invtsq;
    for (u_field::idx t = 100*u.saving_period(); t <= T; t += u.saving_period())
    {
        std::vector<double> ivec;
        std::vector<double> ref;
        std::vector<u_field::real_t> u_back;
        COPY(u.cbegin(t), u.cend(t), std::back_inserter(u_back));
        //thrust::copy(u.cbegin_y(t), u.cend_y(t), std::ostream_iterator<double>(std::cout, ","));
        //std::cout << std::endl; 
        for (unsigned i = 0; i != u.cend(t) - u.cbegin(t); ++i)
        {
            ivec.push_back(i);
            ref.push_back(std::exp(-(2*i*u.gamma0())));
        }
        // plt::title("$u_t[i]$ for fixed $t=" + std::to_string(t) + "$");
        // plt::semilogy(ivec, u_back, "");
        // // plt::semilogy(ivec, ref, "");
        // plt::xlabel("i");
        // plt::ylabel("u");
        // //plt::save("u_plots/t" + std::to_string(t) + ".png");
        // plt::show();

        tvec.push_back(t);
        invtsq.push_back(1./std::sqrt(t));
        mt.push_back(u.avg_R(t) + 3./2./u.gamma0()*std::log(t) - u.velocity() * t);
    }
    //std::cout << "v = " << linear_fit(std::vector<double>(tvec.end()-5, tvec.end()), std::vector<double>(mt.end()-5, mt.end())).first << std::endl;
    
    auto kd = linear_fit(invtsq, mt);
    // std::cout << "1/sqrt(t): [";
    // std::copy(invtsq.begin(), invtsq.end(), std::ostream_iterator<double>(std::cout, ", "));
    // std::cout << "\b\b]" << std::endl;
    // std::cout << "mt-log-vt: [";
    // std::copy(mt.begin(), mt.end(), std::ostream_iterator<double>(std::cout, ", "));
    // std::cout << "\b\b]" << std::endl;
    std::cout << kd.second << " + t^{-1/2} * " << kd.first << std::endl;
    plt::title("Velocity measurement");
    plt::xlabel("$1/\\sqrt{t}$");
    plt::ylabel("$\\langle R_t\\rangle+\\frac{3}{2\\gamma_0}\\log{t}-vt$");
    plt::plot(invtsq, mt, ".");
    plt::plot(invtsq, [&kd](double one_over_sqrt_t){ return kd.first * one_over_sqrt_t + kd.second;}, "-");
    plt::tight_layout();
    plt::show();*/

    return 0;
}