#include <iostream>
#include <iterator>
#include <sstream>
#include <fstream>
#include <thread>

#include "cubrw.hpp"

// #include "matplotlibcpp.h"

// namespace plt = matplotlibcpp;

void evolve_samples(std::vector<branching_random_walk>& samples, int nsteps)
{
    for (branching_random_walk& sample : samples) sample.evolve(nsteps, 1000);
}

int main()
{
    // //benchmark multinomial_distribution
    // std::default_random_engine engine(10);
    // std::cout << "[";
    // for (unsigned N = 1; N < 1000000; N *= 2) std::cout << N << ", ";
    // std::cout << "\b\b], [" << std::flush;
    // for (unsigned N = 1; N < 1000000; N *= 2)
    // {
    //     multinomial_distribution<long> mdist(N, {0.2, 0.3, 0.3, 0.8});
    //     std::vector<double> avg(4);
    //     unsigned N_trials = 1000000;
    //     timer clock;
    //     for (unsigned i = 0; i != N_trials; ++i)
    //     {
    //         auto result = mdist(engine);
    //         avg[0] += result[0] * 1. / N_trials;
    //         avg[1] += result[1] * 1. / N_trials;
    //         avg[2] += result[2] * 1. / N_trials;
    //         avg[3] += result[3] * 1. / N_trials;
    //     }
    //     //std::copy(std::begin(avg), std::end(avg), std::ostream_iterator<int>(std::cout, " "));
    //     std::cout << clock.time() << ", " << std::flush;
    // }
    // std::cout << "\b\b]" << std::endl;

    bool input_from_file = true;
    std::ifstream input_file("config.txt");
    std::istream& input = input_from_file ? input_file : std::cin;
    std::ofstream all_events("all.txt");
    std::cout << "Branching random walk generator.\nTo run a simulation, please provide the following parameters:" << std::endl;
    std::cout << "Number of cores = " << std::flush;
    int nthreads = 3;
    input >> nthreads;
    if (input_from_file) std::cout << nthreads << std::endl;
    int events_per_thread = 1;
    std::cout << "Number of events per thread = " << std::flush;
    input >> events_per_thread;
    if (input_from_file) std::cout << events_per_thread << std::endl;
    int T = 10000;
    std::cout << "Number of timesteps to evolve = " << std::flush;
    input >> T;
    if (input_from_file) std::cout << T << std::endl;
    std::cout << "X-m_T = " << std::flush;
    int X_minus_mT;
    input >> X_minus_mT;
    if (input_from_file) std::cout << X_minus_mT << std::endl;
    std::cout << "Space separated list of values of Δ (distances of yellow particles from the red one(s)): " << std::flush;
    std::string y_values_str;
    getline(input >> std::ws, y_values_str);
    std::stringstream sstr(y_values_str);
    std::vector<int> y_values;
    while (sstr >> y_values_str)
    {
        y_values.push_back(std::stoi(y_values_str));
        if (input_from_file) std::cout << y_values_str << " ";
    }
    if (input_from_file) std::cout << std::endl;
    std::sort(y_values.begin(), y_values.end());
    unsigned device_size = 3000;
    std::cout << "Number of sites to keep track of at the same time (should at least be ~3*sqrt(T)) = " << std::flush;
    input >> device_size;
    if (input_from_file) std::cout << device_size << std::endl;
    std::cout << "Number of timesteps between two checkpoints (not all values of u and y can be saved at the same time) = " << std::flush;
    unsigned saving_period = 300;
    input >> saving_period;
    if (input_from_file) std::cout << saving_period << std::endl;
    double precision = 1e-7;
    std::cout << "Lambda (between 0 and 1) = " << std::flush;
    double lambda;
    input >> lambda;
    if (input_from_file) std::cout << lambda << std::endl;
    if (lambda < 0. || lambda > 1.) throw std::range_error("lambda must be in [0,1]");
    u_field u(int(1000000 * lambda), y_values.begin(), y_values.end(), saving_period);
    int last_checkpoint = (T/saving_period) * saving_period;
    int mT = ((int)(T*u.velocity()-3./2./u.gamma0()*std::log(T))+35)/2*2;
    int X = mT + X_minus_mT;
    unsigned seed;
    std::cout << "Random seed for first event = " << std::flush;
    input >> seed;
    if (input_from_file) std::cout << seed << std::endl;
    std::cout << "Events with seeds from " << seed << " to " << seed + events_per_thread * nthreads - 1 << " will be generated. Press enter to start.";
    if (!input_from_file) input.ignore().get(); //waits for enter key
    std::ofstream output("seeds_" + std::to_string(seed) + "_to_" + std::to_string(seed + events_per_thread * nthreads - 1) + ".csv");
    u.fill_checkpoints(last_checkpoint, device_size, precision);
    u.fill_between(last_checkpoint+1, T+1, device_size, precision);
    std::vector<std::vector<branching_random_walk>> samples(nthreads);
    for (int ithread = 0; ithread != nthreads; ++ithread)
    {
        for (int ievent = 0; ievent != events_per_thread; ++ievent)
        {
            samples[ithread].push_back(branching_random_walk(&u, seed + ievent + ithread * events_per_thread, X, T));
        }
    }
    image ppm(1500, 1000, 50);
    progress_monitor pm;
    timer clock;
    for (int rt = T; rt > 0; rt -= saving_period)
    {
        if (rt != T || last_checkpoint == T)
        {
            u.fill_between(rt-saving_period+1, rt, device_size, precision);
            u.erase(rt+1,rt+saving_period);
            pm.add_datapoint((T-rt)*1./T);
            std::cout << "\x1B[2J\x1B[H" << "Generating events for lambda = " << d_to_str(u.lambda(), 6) << " and T = " << T << "\n" << pm << std::endl;
            std::cout << "Time passed: " << clock << std::endl;
            double max_n = 0.;
            // if (brw.cbegin_y() != brw.cend_y()) max_n = std::max_element(brw.cbegin_y(), brw.cend_y(), [](auto x, auto y){ return x.second < y.second; })->second;
            // std::cout << "Highest particle number on one site: " << max_n << std::endl;
        }

        double vertical_scale_factor = 4.;
        for (auto it = samples[0][0].cbegin_y(); it != samples[0][0].cend_y(); ++it)
        {
            unsigned row = (-it->first + T / vertical_scale_factor) / (2. * T) * (ppm.height() - 1) * vertical_scale_factor;
            unsigned col = (T-rt) / ((double) T) * (ppm.width() - 1);
            if (row+1 < ppm.height())
            {
                ppm.pixel(row, col, {255, 255, 0});
                ppm.pixel(row+1, col, {255, 255, 0});
            }
        }
        for (auto it = samples[0][0].cbegin(); it != samples[0][0].cend(); ++it)
        {
            unsigned row = (-*it + T / vertical_scale_factor) / (2. * T) * (ppm.height() - 1) * vertical_scale_factor;
            unsigned col = (T-rt) / ((double) T) * (ppm.width() - 1);
            if (row+1 < ppm.height())
            {
                ppm.pixel(row, col, {255, 0, 0});
                ppm.pixel(row+1, col, {255, 0, 0});
            }
        }
        unsigned evolution_steps;
        if (rt != T || last_checkpoint == T) evolution_steps = saving_period;
        else
        {
            evolution_steps = T-last_checkpoint;
            rt = last_checkpoint + saving_period;
        }
        std::vector<std::thread> threads;
        for (int ithread = 0; ithread != nthreads; ++ithread) threads.push_back(std::thread(evolve_samples, std::ref(samples[ithread]), evolution_steps));
        for (int ithread = 0; ithread != nthreads; ++ithread) threads[ithread].join();
    }
    std::cout << "It took " << clock << "." << std::endl;
    ppm.save("output.ppm");
    // std::cout << "Distribution of yellow particles: ";
    // for (auto it = brw.cbegin_y(); it != brw.cend_y(); ++it)
    // {
    //     std::cout << "(" << it->first << "; " << it->second << ") ";
    // }
    // std::cout << std::endl;
    output << X;
    for (auto it = y_values.begin(); it != y_values.end(); ++it) output << ", " << (X-*it);
    output << std::endl;
    int n_rejected = 0;
    for (int ithread = 0; ithread != nthreads; ++ithread)
    {
        for (int ievent = 0; ievent != events_per_thread; ++ievent)
        {
            int nred = std::count(samples[ithread][ievent].cbegin(), samples[ithread][ievent].cend(), X);
            if (nred == 0)
            {
                ++n_rejected;
                all_events << "Red particle did not arrive at final position." << std::endl;
                continue;
            }
            else output << nred;
            std::copy(samples[ithread][ievent].cbegin(), samples[ithread][ievent].cend(), std::ostream_iterator<int>(all_events, ", "));
            for (auto it = samples[ithread][ievent].cbegin_y(); it != samples[ithread][ievent].cend_y(); ++it) all_events << "(" << it->first << ", " << it->second << ") ";
            all_events << std::endl;
            for (auto it = y_values.begin(); it != y_values.end(); ++it)
            {
                output << ", " << samples[ithread][ievent].y_at(X-*it);
            }
            output << std::endl;
        }
    }
    std::cout << "Rejected " << n_rejected << " events." << std::endl;
    std::cout << "Number of calls to binomial_distribution = " << multinomial_distribution<int>::n_calls_to_binom_dist << std::endl;
    output.close();
    all_events.close();
    return 0;
}