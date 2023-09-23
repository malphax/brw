#include "brw.hpp"
#include <functional>
#include <iostream>
#include <iomanip>

namespace brw
{
    std::string d_to_str(double x, int precision)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(precision) << x;
        return ss.str();
    }

    /////////////////
    ////stopwatch////
    /////////////////

    std::ostream& operator<<(std::ostream& out, const stopwatch& s)
    {
        auto time = s.time();
        out << time / 3600000 << " h " << (time % 3600000) / 60000 << " min " << (time % 60000) / 1000 << " s " << time % 1000 << " ms";
        return out;
    }

    ////////////////////////
    ////progress_monitor////
    ////////////////////////

    std::pair<double, double> progress_monitor::linear_fit(const std::vector<double>& x, const std::vector<double>& y) const
    {
        auto x_it = x.begin();
        auto y_it = y.begin();
        double sum_xy = 0;
        double sum_xx = 0;
        double sum_x = 0;
        double sum_y = 0;
        unsigned n = x.size();
        while (y_it != y.end() && x_it != x.end())
        {
            sum_xy += *x_it * *y_it;
            sum_xx += *x_it * *x_it;
            sum_x += *x_it;
            sum_y += *y_it;
            ++y_it;
            ++x_it;
        }
        double k = (sum_xy - sum_x * sum_y / n) / (sum_xx - sum_x * sum_x / n);
        double d = (sum_y - k * sum_x) / n;
        return {k,d};
    }

    void progress_monitor::reset()
    {
        starting_time = std::chrono::steady_clock::now();
        progress.erase(progress.begin(), progress.end());
        times.erase(times.begin(), times.end());
    }

    void progress_monitor::add_datapoint(double percent_progress)
    {
        auto now = std::chrono::steady_clock::now();
        unsigned long diff = std::chrono::duration_cast<std::chrono::milliseconds>(now-starting_time).count();
        progress.push_back(percent_progress);
        times.push_back(diff);
        if (progress.size() > size)
        {
            progress.pop_front();
            times.pop_front();
        }
    }

    unsigned progress_monitor::time_remaining() const
    {
        std::vector<double> times_vec(times.begin(), times.end());
        std::vector<double> progress_vec(progress.begin(), progress.end());
        auto kd = linear_fit(times_vec, progress_vec);
        return (1. - kd.second) / kd.first - times.back();
    }

    std::ostream& operator<<(std::ostream& out, const progress_monitor& pm)
    {
        auto time_left = pm.time_remaining() / 1000; //in seconds
        out << (int) (pm.progress.back()*100) << "%. Remaining time: " << time_left / 3600 << " h " << (time_left % 3600) / 60 << " min " << time_left%60 << " s.";
        return out;
    }

    ///////////////////
    ////prob_fields////
    ///////////////////

    prob_fields::prob_fields(unsigned lambda_pµ, unsigned saving_period, unsigned device_size) : _lambda_pµ(lambda_pµ), _lambda(lambda_pµ / 1e6), _saving_period(saving_period), _compute_y(false), device_size(device_size)
    {
        if (lambda_pµ > 1000000) throw std::range_error("Out of range: lambda_pµ must be within [1,1e6].");
        compute_velocity();
    }

    prob_fields::real_t prob_fields::u(idx t, sidx x) const
    {
        sidx st = t;
        if (t > std::numeric_limits<sidx>::max()) throw std::domain_error("t is to large to be converted to a signed number.");
        if (x < -st) return 1.;
        else if (x > st) return 0.;
        return u_ti(t, (t + x + 1)/2); //if 2|(t+x), the +1 is rounded away, otherwise u(x,t)=u(x+1,t)
    }

    prob_fields::real_t prob_fields::y(idx t, sidx x) const
    {
        if (!_compute_y) throw std::runtime_error("y has not been computed, please use a different constructor!");
        sidx st = t;
        if (t > std::numeric_limits<sidx>::max()) throw std::domain_error("t is to large to be converted to a signed number.");
        if (x < -st) return 0.;
        return ((x+t)%2==0) ? y_ti(t, (t + x)/2) : 0.;
    }

    void prob_fields::fill_checkpoints(idx T, double tol)
    {
        progress_monitor pm(20);
        idx output_period = T > 1000 ? 1000 : 1;
        stopwatch clock;

        idx t_max = T - T%_saving_period;
        idx t_min = (u_map.size() == 0) ? 0 : u_map.rbegin()->first + 1;

        if (t_min == 0)
        {
            u_map[0].first = 0;
            u_map[0].second = std::vector<real_t>(device_size, 0);
            u_map[0].second[0] = 1;
            ++t_min;
            if (device_size < y_map[0].size()) throw std::runtime_error("device_size must be chosen large enough to contain all of the destinations of yellow particles");
            else y_map[0].resize(device_size);
        }
        lsrb = u_map[t_min-1].first;
        prev_u = u_map.at(t_min-1).second;
        prev_y = y_map.at(t_min-1);
        for (idx t = t_min; t != t_max + 1; ++t)
        {
            compute_row(t, tol, t % _saving_period == 0);
            if ((t + 1) % output_period == 0)
            {
                pm.add_datapoint((t - (double) t_min) / (double) t_max);
                std::cout << "\x1B[2J\x1B[H" << "Filling u for lambda = " << d_to_str(_lambda, 6) << " for T = " << T << "\n" << pm << std::endl;
                std::cout << "Approximate memory usage: " << estimate_memory() / (1 << 20) << " MB" << std::endl;
            }
        }
        std::cout << "\x1B[2J\x1B[H" << "Filled u for lambda = " << d_to_str(_lambda, 6) << " for T = " << T << std::endl;
        std::cout << "It took " << clock << "." << std::endl;
        std::cout << "Approximate memory usage: " << estimate_memory() / (1 << 20) << " MB" << std::endl;
    }

    void prob_fields::fill_between(idx T1, idx T2, double tol)
    {
        if (T1 == 0)
        {
            u_map[0].first = 0;
            u_map[0].second = std::vector<real_t>(device_size, 0);
            u_map[0].second[0] = 1;
            ++T1;
            if (device_size < y_map[0].size()) throw std::runtime_error("device_size must be chosen large enough to contain all of the destinations of yellow particles");
            else y_map[0].resize(device_size);
        }
        lsrb = u_map[T1-1].first;
        prev_u = u_map.at(T1-1).second;
        prev_y = y_map.at(T1-1);
        for (idx t = T1; t != T2; ++t) compute_row(t, tol, true);
    }

    unsigned long prob_fields::estimate_memory() const
    {
        if (u_map.size() == 0 || u_map.rbegin()->second.second.size() == 0) return 0;
        else return (2 * sizeof(idx) + sizeof(real_t) * u_map.rbegin()->second.second.size()) * u_map.size();
    }

    void prob_fields::compute_velocity()
    {
        if (_lambda_pµ == 0)
        {
            _velocity = 0.;
            _gamma0 = 0.;
            return;
        }
        else if (_lambda_pµ == 1000000U)
        {
            _velocity = 1.;
            _gamma0 = std::numeric_limits<real_t>::max();
            return;
        }
        real_t gamma1 = sqrt(2.*log(1.+_lambda));
        std::function<real_t(real_t)> f = [this](real_t g){ return g*std::tanh(g)-std::log((1.+_lambda)*std::cosh(g)); };
        std::function<real_t(real_t)> f_prime = [this](real_t g){ return g*(1-std::pow(std::tanh(g),2)); };
        unsigned i = 0;
        unsigned max_steps = 300;
        do
        {
            _gamma0 = gamma1;
            gamma1 = _gamma0 - f(_gamma0) / f_prime(_gamma0);
            ++i;
        }
        while (std::abs(gamma1 - _gamma0)/_gamma0 > 1e-12 && i != max_steps);
        _velocity = std::log((1.+_lambda)*std::cosh(gamma1)) / gamma1;
        _gamma0 = gamma1;
        if (i == max_steps) throw std::domain_error("Newton's method did not reach the required degree of precision within " + std::to_string(max_steps) + " steps.");
    }

    void prob_fields::compute_row(idx t, double tol, bool save)
    {
        next_u.clear();
        real_t newfront = u_recursion(*prev_u.begin(), 1.);

        bool shift = (newfront >= 1 - tol);

        if (!shift) next_u.push_back(newfront);
        else ++lsrb;

        for (auto it = prev_u.begin(); it != prev_u.end()-1; ++it) next_u.push_back(u_recursion(*it, *(it+1)));

        if (shift) next_u.push_back(u_recursion(*(prev_u.end()-1), 0));

        if (_compute_y)
        {
            next_y.clear();
            real_t newfront_y = y_recursion(*prev_u.begin(), 1., *prev_y.begin(), 0.);

            /* Potential design flaw:
             * Whether newfront_y is kept only depends on the newfront(_u). Thus, while the accuracy in u is controlled,
             * the accuracy of y is undefined.
             */
            if (!shift) next_y.push_back(newfront_y);

            auto y_it = prev_y.begin();
            for (auto it = prev_u.begin(); it != prev_u.end()-1; ++it)
            {
                next_y.push_back(y_recursion(*it, *(it+1), *y_it, *(y_it+1)));
                ++y_it;
            }

            if (shift) next_y.push_back(y_recursion(*(prev_u.end()-1), 0, *(prev_y.end()-1), 0));

            if (save) y_map[t] = next_y;
            std::swap(prev_y, next_y);
        }

        if (save) 
        {
            u_map[t].first = lsrb;
            u_map[t].second = next_u;
        }
        std::swap(prev_u, next_u);        
    }

    prob_fields::real_t prob_fields::u_ti(idx t, idx i) const
    {
        if (i == 0 || i < lower_scaling_region_bound(t)) return 1.;
        else if (i >= upper_scaling_region_bound(t)) return 0.;
        else return u_map.at(t).second[i - lower_scaling_region_bound(t)]; //like this, only rows in u that have been filled can be accessed.
    }

    prob_fields::real_t prob_fields::y_ti(idx t, idx i) const
    {
        if (!_compute_y) throw std::runtime_error("y has not been computed, please use a different constructor!");
        if (i == 0 || i < lower_scaling_region_bound(t)) return 0.;
        else if (i >= upper_scaling_region_bound(t)) return 0.;
        else return y_map.at(t)[i - lower_scaling_region_bound(t)]; //like this, only rows in u that have been filled can be accessed.
    }
}