#include <iostream>
#include <random>
#include <fstream>
#include <vector>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

std::vector<long double> bessel_process(unsigned n_steps, long double end_time, unsigned seed)
{
    std::mt19937 engine(seed);
    std::uniform_int_distribution<> dist(1,6);
    std::vector<long double> result = { 0. };
    if (n_steps == 0) return result;
    unsigned x = 0, y = 0, z = 0;
    long double step_size = std::sqrt(end_time / n_steps);
    for (unsigned t = 0; t != n_steps; ++t)
    {
        int direction = dist(engine);
        switch (direction)
        {
            case 1:
                ++x; break;
            case 2:
                --x; break;
            case 3:
                ++y; break;
            case 4:
                --y; break;
            case 5:
                ++z; break;
            case 6:
                --z; break;
            default:
                throw std::range_error("Direction code was not within 1 to 6");
        }
        result.push_back(std::hypot(x,y,z) / step_size);
    }
    return result;
}

class integral
{
    private:
        long double a,b,c;
        std::vector<long double> sample_process;
        std::vector<long double> sample_integrand;
        long double integral_value;
        long double value_at_half_time_steps;
        long double time_step_size;
        long double space_step_size;
    public:
        //Approximately computes the integral \int_{t=0}^\infty dt \exp(-\alpha r_t-\beta / t + c) for an exemplary Bessel process r_t
        integral(long double alpha, long double beta, long double gamma, long double time_step_size = 0.01) : a(alpha), b(beta), c(gamma), time_step_size(time_step_size), space_step_size(std::sqrt(time_step_size)) {}
        //Returns the value of the integral corresponding to its last evaluation
        long double value() const { return integral_value; }
        //Returns the value of the integral corresponding to its last evaluation when only half the sampling points are used.
        long double value_half_precision() const { return value_at_half_time_steps; }
        /*Performs a saddle point approximation where the saddle point t_s is chosen such that |(2*\beta/\alpha/r_t*sqrt(t_s))^{2/3}-t_s| is minimized.
        The integral must be evaluated beforehand, so that a sample process is present.*/
        long double naive_saddle_point_approx() const {
            long double rts;
            long double ts;
            long double min_error = 1e100;
            for (unsigned t = 0; t != sample_process.size(); ++t)
            {
                long double t_double = t * time_step_size;
                long double curr_error = std::abs(std::pow(2*b/a/sample_process[t]*std::sqrt(t_double), 2./3) - t_double);
                if (curr_error < min_error)
                {
                    ts = t_double;
                    rts = sample_process[t];
                    min_error = curr_error;
                }
            }
            //std::cout << "min_error = " << min_error << std::endl;
            // std::cout << "S(ts) = " << a * rts + b/ts - c << std::endl;
            return std::sqrt(M_PI * b / 3.) * 4 / a / rts * std::sqrt(ts) * std::exp(-a * rts - b / ts + c);
        }
        //Performs a saddle point approximation
        long double advanced_saddle_point_approx() const {
            long double rts;
            long double ts;
            long double min = 1e100;
            for (unsigned t = 1; t != sample_process.size(); ++t)
            {
                long double t_double = t * time_step_size;
                long double curr = a * sample_process[t] + b / t_double;
                if (curr < min)
                {
                    ts = t_double;
                    rts = sample_process[t];
                    min = curr;
                }
            }
            // std::cout << "min S = " << min - c << std::endl;
            long double prefactor  = std::sqrt(M_PI * b / 3.) * 4 / a / rts * std::sqrt(ts);
            return std::exp(-a * rts - b / ts + c);
        }
        //Returns the bessel process corresponding to the last evaluation of the integral
        const std::vector<long double>& sample() const { return sample_process; }
        //Returns the integrand corresponding to the last sample
        const std::vector<long double>& integrand() const { return sample_integrand; }
        //Generates a process and evaluates the integral.
        long double evaluate(unsigned seed, long double tol = 1e-10, unsigned max_steps = 1 << 20)
        {
            std::mt19937 engine(seed);
            std::uniform_int_distribution<> dist(1,6);
            unsigned min_steps = 2 * std::pow(b/a, 2./3.) / time_step_size;
            sample_process.clear();
            sample_process.push_back(0.);
            sample_integrand.clear();
            sample_integrand.push_back(0.);
            integral_value = 0.;
            value_at_half_time_steps = 0.;
            bool even = false;
            int x = 0, y = 0, z = 0;
            for (unsigned t = 1; t != max_steps; ++t)
            {
                int direction = dist(engine);
                switch (direction)
                {
                    case 1:
                        ++x; break;
                    case 2:
                        --x; break;
                    case 3:
                        ++y; break;
                    case 4:
                        --y; break;
                    case 5:
                        ++z; break;
                    case 6:
                        --z; break;
                    default:
                        throw std::range_error("Direction code was not within 1 to 6");
                }
                sample_process.push_back(std::hypot(x,y,z) * space_step_size);
                sample_integrand.push_back(std::exp(-a * sample_process.back() - b / (t * time_step_size) + c));
                long double change = time_step_size * sample_integrand.back();
                integral_value += change;
                if (even) value_at_half_time_steps += 2 * change; //use only half the values but weight them long double
                even = !even;
                if (change / integral_value < tol && t > min_steps) return integral_value;
            }
            std::cout << "Warning: The maximal number of steps was reached for seed " << seed << ", tolerance " << tol << "." << std::endl;
            return integral_value;
        }
};

int main()
{
    long double alpha = 1.;
    long double beta = 1000.;
    //std::ofstream file("bessel.out");
    std::vector<long double> samples;
    std::vector<long double> approximated;

    unsigned n_samples = 10000;
    for (unsigned i = 0; i != n_samples; ++i)
    {
        if (i % 50 == 1) std::cout << "\x1B[2J\x1B[HProgress: " << int(i*100./n_samples) << "%." << std::endl;
        integral I(alpha, beta, 0., 1e-2);
        I.evaluate(i, 1e-7);
        samples.push_back(- (std::log(I.value())) / std::cbrt(beta));
        approximated.push_back(- (std::log(I.naive_saddle_point_approx())) / std::cbrt(beta));
        
        // std::cout << "Integral value (" << I.sample().size() << " sampling points; seed " << i << "): " << I.value();
        // std::cout << " (" << I.value_half_precision() - I.value() << ")\nNaive saddle point approximation: " << I.naive_saddle_point_approx() << std::endl;
        // std::cout << "Better saddle point approximation: " << I.advanced_saddle_point_approx() << std::endl;
        // plt::plot(I.integrand());
        // plt::show();
    }
    std::vector<long double> formula;
    std::vector<long double> l_vec;
    unsigned n_bins = 20;
    long double max = *std::max_element(samples.begin(), samples.end());
    long double min = *std::min_element(samples.begin(), samples.end());
    for (long double l = 0; l < max + 10 ; l += 0.1)
    {
        l_vec.push_back(l);
        formula.push_back(samples.size()*(max-min)/n_bins*16./27/std::sqrt(3 * M_PI)/std::pow(alpha,3)*std::pow(l, 7./2.)*std::exp(-4./27./std::pow(alpha,2)*std::pow(l, 3.)));
    }

    plt::hist(samples, n_bins);
    plt::hist(approximated, n_bins, "r", 0.5);
    plt::plot(l_vec, formula);
    plt::show();

    return 0;
}