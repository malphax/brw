#define GPU_SUPPORT

#include "cubrw.hpp"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/for_each.h>
#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <iterator>

///////////////////////////////////////////////// BRW /////////////////////////////////////////////////

std::mt19937 BRW::engine(BRW::seed);

void BRW::evolve_one_step(unsigned long det_thr)
{
    std::map<long, particle_number> new_n;
    for (auto it = n.cbegin(); it != n.cend(); ++it)
    {
        if (it->second > det_thr)
        {
            particle_number n_plusminus = it->second*(1.+lam)/2.;
            new_n[it->first - 1] += n_plusminus;
            new_n[it->first + 1] += n_plusminus;
            continue;
        }
        particle_number n_minus = std::binomial_distribution<particle_number>(it->second)(engine);
        particle_number n_plus = it->second - n_minus;
        n_minus += std::binomial_distribution<particle_number>(n_minus, lam)(engine);
        n_plus += std::binomial_distribution<particle_number>(n_plus, lam)(engine);

        if (n_minus != 0) new_n[it->first - 1] += n_minus;
        if (n_plus != 0) new_n[it->first + 1] += n_plus;
    }
    n.swap(new_n);
    rightmost.push_back(n.rbegin()->first);
}

void BRW::evolve(long n_steps, unsigned long det_thr)
{
    for (unsigned i = 0; i != n_steps; ++i)
    {
        evolve_one_step(det_thr);
        ++time;
    }
}

std::string d_to_str(double x, int precision)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << x;
    return ss.str();
}

///////////////////////////////////////////////// progress_monitor /////////////////////////////////////////////////

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
    auto kd = linear_fit<double>(times_vec, progress_vec);
    return (1. - kd.second) / kd.first - times.back();
}

std::ostream& operator<<(std::ostream& out, const progress_monitor& pm)
{
    auto time_left = pm.time_remaining() / 1000; //in seconds
    out << (int) (pm.progress.back()*100) << "%. Remaining time: " << time_left / 3600 << " h " << (time_left % 3600) / 60 << " min " << time_left%60 << " s.";
    return out;
}

//Prints time since last reset.
std::ostream& operator<<(std::ostream& out, const timer& t)
{
    auto time = t.time();
    out << time / 3600000 << " h " << (time % 3600000) / 60000 << " min " << (time % 60000) / 1000 << " s " << time % 1000 << " ms";
    return out;
}

///////////////////////////////////////////////// u_field /////////////////////////////////////////////////

u_field::u_field(unsigned lambda_pµ, unsigned saving_period) : _lambda_pµ(lambda_pµ), _lambda(lambda_pµ / 1e6), _saving_period(saving_period), _compute_y(false)
{
    if (lambda_pµ > 1000000) throw std::range_error("Out of range: lambda_pµ must be within [1,1e6].");
    compute_velocity();
}

u_field::u_field(unsigned lambda_pµ, std::initializer_list<int> x_y0, unsigned saving_period) : u_field(lambda_pµ, saving_period)
{
    _compute_y = true;
    if (x_y0.size() == 0) throw std::runtime_error("The initializer_list<int> x_y0 must be non-empty if it is provided.");
    int xmin = std::min(x_y0);
    int xmax = std::max(x_y0);
    if (xmin <= 0) throw std::domain_error("The sites x at which y(0,x) != 0 must all be strictly positive.");
    for (auto x : x_y0)
    {
        if (x % 2 == 0)
        {
            y0.resize(x/2 + 1);
            y0[x/2] = 1;
        }
    }
}

u_field::real_t u_field::operator()(idx t, sidx x) const
{
    sidx st = t;
    if (t > std::numeric_limits<sidx>::max()) throw std::domain_error("t is to large to be converted to a signed number.");
    if (x < -st) return 1.;
    else if (x > st) return 0.;
    return u_ti(t, (t + x + 1)/2); //if 2|(t+x), the +1 is rounded away, otherwise u(x,t)=u(x+1,t)
}

u_field::real_t u_field::y(idx t, sidx x) const
{
    sidx st = t;
    if (t > std::numeric_limits<sidx>::max()) throw std::domain_error("t is to large to be converted to a signed number.");
    if (x < -st) return 0.;
    return ((x+t)%2==0) ? y_ti(t, (t + x)/2) : 0.;
}

void u_field::fill_checkpoints(idx T, unsigned device_size, double tol)
{
    progress_monitor pm(20);
    idx output_period = T > 1000 ? 1000 : 1;
    timer clock;

    idx t_max = T - T%_saving_period;
    idx t_min = (u_map.size() == 0) ? 0 : u_map.rbegin()->first + 1;

    prev_u = thrust::device_vector<real_t>(device_size, 0.);
    next_u = thrust::device_vector<real_t>(device_size, 0.);
    prev_y = thrust::device_vector<real_t>(device_size, 0.);
    next_y = thrust::device_vector<real_t>(device_size, 0.);

    if (t_min != 0)
    {

        thrust::copy(u_map[t_min-1].second.cbegin(), u_map[t_min-1].second.cend(), prev_u.begin());
        if (_compute_y) thrust::copy(y_map[t_min-1].cbegin(), y_map[t_min-1].cend(), prev_y.begin());
        lsrb = u_map[t_min-1].first;
    }
    else
    {
        thrust::copy(y0.cbegin(), y0.cend(), prev_y.begin());
        y_map[0] = y0;
        prev_u[0] = 1.;
        u_map[0].second = prev_u;
        u_map[0].first = 0;
        ++t_min;
        lsrb = 0;
    }

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

void u_field::fill_between(idx T1, idx T2, unsigned device_size, double tol)
{
    prev_u = thrust::device_vector<real_t>(device_size, 0.);
    next_u = thrust::device_vector<real_t>(device_size, 0.);
    prev_y = thrust::device_vector<real_t>(device_size, 0.);
    next_y = thrust::device_vector<real_t>(device_size, 0.);

    if (T1 != 0)
    {
        thrust::copy(u_map[T1-1].second.begin(), u_map[T1-1].second.end(), prev_u.begin());
        if (_compute_y) thrust::copy(y_map[T1-1].cbegin(), y_map[T1-1].cend(), prev_y.begin());
        lsrb = u_map[T1-1].first;
    }
    else
    {
        thrust::copy(y0.cbegin(), y0.cend(), prev_y.begin());
        y_map[0] = y0;
        prev_u[0] = 1.;
        u_map[0].second = prev_u;
        u_map[0].first = 0;
        ++T1;
        lsrb = 0;
    }

    for (idx t = T1; t != T2; ++t) compute_row(t, tol, true);
}

std::ostream& operator<<(std::ostream& out, const u_field& u)
{
    out << u._lambda_pµ << " " << u._saving_period << std::endl;
    for (auto &[key,value]: u.u_map)
    {
        out << key << " "; //time
        out << value.first << " "; //lower_scaling_region_bound;
        out << std::hexfloat;
        thrust::copy(value.second.begin(), value.second.end(), std::ostream_iterator<u_field::real_t>(out, " "));
        out << std::endl;
    }
    return out;
}

std::istream& operator>>(std::istream& in, u_field& u)
{
    double lam_pµ;
    unsigned sav_per;
    
    if (!(in >> lam_pµ)) throw std::runtime_error("Couldn't read in lambda.");
    u._lambda_pµ = lam_pµ;
    u._lambda = lam_pµ / 1e6;
    
    if (!(in >> sav_per)) throw std::runtime_error("Couldn't read in saving_period.");
    u._saving_period = sav_per;
        
    std::string line;
    u.u_map.clear();
    if(!std::getline(in >> std::ws, line)) throw std::runtime_error("Now values of u to read in.");
    do
    {
        std::istringstream iss(line);
        u_field::idx t; //time and lower_scaling_region_bound
        if (!(iss >> t)) throw std::runtime_error("Couldn't read in timestep.");
        else if (!(iss >> u.u_map[t].first)) throw std::runtime_error("Couldn't read in lower_approx_bound.");
        std::string hexvalue;
        while (iss >> hexvalue) u.u_map[t].second.push_back(std::strtold(hexvalue.c_str(), NULL));
    }
    while (std::getline(in, line));

    return in;
}

unsigned long u_field::estimate_memory(bool rough) const
{
    if (u_map.size() == 0 || u_map.rbegin()->second.second.size() == 0) return 0;
    else if (rough) return (2 * sizeof(idx) + sizeof(real_t) * u_map.rbegin()->second.second.size()) * u_map.size();
    unsigned long num_vec_entries = 0;
    for (const auto& kd : u_map) num_vec_entries += kd.second.second.size();
    return sizeof(idx) * u_map.size() + num_vec_entries * sizeof(real_t) * (_compute_y ? 2 : 1);
}

void u_field::print(std::ostream& out, unsigned digits, idx window_size)
{
    for (sidx t = 0; t != window_size; ++t)
    {
        for (sidx x = - (sidx) window_size; x != window_size + 1; ++x)
        {
            out << std::setw(digits + 3) << d_to_str((*this)(t,x), digits);
        }
        out << "\n" << std::endl;
    }
}

void u_field::compute_velocity()
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

void u_field::compute_row(idx t, double tol, bool save)
{
    auto recursion = u_recursion(_lambda);

    auto beg_u = prev_u.cbegin();
    auto end_u = prev_u.cend();
    decltype(next_u.begin()) beg_fill_u;
    decltype(next_u.end()) end_fill_u;
    auto beg_y = prev_y.cbegin();
    auto end_y = prev_y.cend();
    decltype(next_y.begin()) beg_fill_y;
    decltype(next_y.end()) end_fill_y;
    real_t newfront, newfront_y;
    recursion(newfront, prev_u.front(), 1., newfront_y, prev_y.front(), 0.);
    if (1. - newfront < tol)
    {
        ++lsrb;
        beg_fill_u = next_u.begin();
        end_fill_u = next_u.end()-1;
        beg_fill_y = next_y.begin();
        end_fill_y = next_y.end()-1;
        //next_u.back() = recursion(prev_u.back(), 0.);
    }
    else
    {
        if (_compute_y) next_y[0] = newfront_y;
        next_u[0] = newfront;
        beg_fill_u = next_u.begin() + 1;
        end_fill_u = next_u.end();
        beg_fill_y = next_y.begin() + 1;
        end_fill_y = next_y.end();
        --end_u;
        --end_y;
    }
    
    if (_compute_y)
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(beg_fill_u, beg_u, beg_u+1, beg_fill_y, beg_y, beg_y+1)),
                         thrust::make_zip_iterator(thrust::make_tuple(end_fill_u, end_u-1, end_u, end_fill_y, end_y-1, end_y)),
                         thrust::make_zip_function(recursion));
    else
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(beg_fill_u, beg_u, beg_u+1)),
                         thrust::make_zip_iterator(thrust::make_tuple(end_fill_u, end_u-1, end_u)),
                         thrust::make_zip_function(recursion));
    // thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(beg_fill_y, beg_y, )),
    //                  thrust::make_zip_iterator(thrust::make_tuple()));
    // thrust::transform(beg_u, end_u-1, beg_u+1, beg_fill_u, recursion);

    if (save && _compute_y)
    {
        u_map[t].first = lsrb;
        u_map[t].second = next_u;
        y_map[t] = next_y;
    }
    else if (save)
    {
        u_map[t].first = lsrb;
        u_map[t].second = next_u;
    }
    
    //Prepare for next call to compute_row.
    thrust::swap(prev_u, next_u);
    if (_compute_y) thrust::swap(prev_y, next_y);
}

u_field::real_t u_field::u_ti(idx t, idx i) const
{
    if (i == 0 || i < lower_scaling_region_bound(t)) return 1.;
    else if (i >= upper_scaling_region_bound(t)) return 0.;
    else
    {
        return u_map.at(t).second[i - lower_scaling_region_bound(t)]; //like this, only rows in u that have been filled can be accessed.
        // auto it = u_map.find(t);
        // if (it != u_map.end()) return it->second.at(i - lower_approx_bound[t]);
        // if (i != 0)
        // {
        //     return recursion(u(t-1,i), u(t-1,i-1));
        // }
        // else return 1.;
    }
}

u_field::real_t u_field::y_ti(idx t, idx i) const
{
    if (i == 0 || i < lower_scaling_region_bound(t)) return 0.;
    else if (i >= upper_scaling_region_bound(t)) return 0.;
    else
    {
        return y_map.at(t)[i - lower_scaling_region_bound(t)]; //like this, only rows in u that have been filled can be accessed.
    }
}

///////////////////////////////////////////////// branching_random_walk /////////////////////////////////////////////////

// void branching_random_walk::evolve_one_step(unsigned long det_thr)
// {
//     std::map<int, ptcl_n> new_n;
//     std::map<int, ptcl_n> new_n1;
//     u_field& u = *ptr_u;
//     for (auto [x,num] : n)
//     {
//         double Utilde_ratio_left = (u(T-t-1, X-x+1) - u(T-t-1, X-x+2)) / (u(T-t, X-x) - u(T-t, X-x+1));
//         double Utilde_ratio_right = (u(T-t-1, X-x-1) - u(T-t-1, X-x)) / (u(T-t, X-x) - u(T-t, X-x+1));
//         double p_r_to_r_right  = Utilde_ratio_right * (1.-u.lambda())/2.; //((1.+u.lambda())/2. - u.lambda() * u(T-t-1, X-x-1));
//         double p_r_to_2r_right = Utilde_ratio_right * u.lambda()/2. * (u(T-t-1, X-x-1) - u(T-t-1, X-x));
//         double p_r_to_rb_right = Utilde_ratio_right * u.lambda() * (1. - u(T-t-1, X-x+1));
//         double p_r_to_r_left   = Utilde_ratio_left * (1.-u.lambda())/2.; //((1.+u.lambda())/2. - u.lambda() * u(T-t-1, X-x+1));
//         double p_r_to_2r_left  = Utilde_ratio_left * u.lambda()/2. * (u(T-t-1, X-x+1) - u(T-t-1, X-x+2));
//         double p_r_to_rb_left  = Utilde_ratio_left * u.lambda() * (1. - u(T-t-1, X-x-1));
        
//         auto gains = multinomial_distribution<ptcl_n>(num, { p_r_to_r_right,
//                                                              p_r_to_2r_right,
//                                                              p_r_to_rb_right,
//                                                              p_r_to_r_left,
//                                                              p_r_to_2r_left,
//                                                              p_r_to_rb_left })(engine);
//         ptcl_n gain_right = gains[0] + 2*gains[1] + gains[2];
//         ptcl_n gain_left = gains[3] + 2*gains[4] + gains[5];
//         if (gain_right != 0) new_n[x+1] += gain_right;
//         if (gain_left != 0) new_n[x-1] += gain_left;
//         if (gains[2] != 0) new_n1[x+1] += gains[2];
//         if (gains[5] != 0) new_n1[x-1] += gains[5];
        
//     }
//     for (auto [x,num] : n1)
//     {
//         double p_b_to_b_right  = (1. - u(T-t-1, X-x+1)) / (1. - u(T-t, X-x)) * (1.-u.lambda())/2.;
//         double p_b_to_2b_right = (1. - u(T-t-1, X-x+1)) * (1. - u(T-t-1, X-x+1)) / (1. - u(T-t, X-x)) * u.lambda()/2.;
//         double p_b_to_b_left   = (1. - u(T-t-1, X-x-1)) / (1. - u(T-t, X-x)) * (1.-u.lambda())/2.;
//         double p_b_to_2b_left  = (1. - u(T-t-1, X-x-1)) * (1. - u(T-t-1, X-x-1)) / (1. - u(T-t, X-x)) * u.lambda()/2.;

//         // double p_b_to_b_right  = (1.-u.lambda())/2.;
//         // double p_b_to_2b_right = u.lambda()/2.;
//         // double p_b_to_b_left   = (1.-u.lambda())/2.;
//         // double p_b_to_2b_left  = u.lambda()/2.;

//         auto gains = multinomial_distribution<ptcl_n>(num, { p_b_to_b_right,
//                                                              p_b_to_2b_right,
//                                                              p_b_to_b_left,
//                                                              p_b_to_2b_left })(engine);
//         ptcl_n gain_right = gains[0] + 2*gains[1];
//         ptcl_n gain_left = gains[2] + 2*gains[3];
//         if (gain_right != 0) new_n1[x+1] += gain_right;
//         if (gain_left != 0) new_n1[x-1] += gain_left;
//     }
//     n = std::move(new_n);
//     n1 = std::move(new_n1);
// }

void branching_random_walk::evolve_one_step(unsigned long det_thr)
{
    std::map<int, ptcl_n> new_n;
    std::map<int, ptcl_n> new_n1;
    u_field& u = *ptr_u;
    for (auto [x,num] : n)
    {
        double Utilde_ratio_left = (u(T-t-1, X-x+1) - u(T-t-1, X-x+2)) / (u(T-t, X-x) - u(T-t, X-x+1));
        double Utilde_ratio_right = (u(T-t-1, X-x-1) - u(T-t-1, X-x)) / (u(T-t, X-x) - u(T-t, X-x+1));
        double p_r_to_r_right  = Utilde_ratio_right * ((1.+u.lambda())/2. - u.lambda() * (u(T-t-1, X-x-1) + u.y(T-t-1, X-x-1)));
        double p_r_to_2r_right = Utilde_ratio_right * u.lambda()/2. * (u(T-t-1, X-x-1) - u(T-t-1, X-x));
        double p_r_to_rb_right = Utilde_ratio_right * u.lambda() * u.y(T-t-1, X-x+1);
        double p_r_to_r_left   = Utilde_ratio_left * ((1.+u.lambda())/2. - u.lambda() * (u(T-t-1, X-x+1) + u.y(T-t-1, X-x+1)));
        double p_r_to_2r_left  = Utilde_ratio_left * u.lambda()/2. * (u(T-t-1, X-x+1) - u(T-t-1, X-x+2));
        double p_r_to_rb_left  = Utilde_ratio_left * u.lambda() * u.y(T-t-1, X-x-1);
        
        auto gains = multinomial_distribution<ptcl_n>(num, { p_r_to_r_right,
                                                             p_r_to_2r_right,
                                                             p_r_to_rb_right,
                                                             p_r_to_r_left,
                                                             p_r_to_2r_left,
                                                             p_r_to_rb_left })(engine);
        ptcl_n gain_right = gains[0] + 2*gains[1] + gains[2];
        ptcl_n gain_left = gains[3] + 2*gains[4] + gains[5];
        if (gain_right != 0) new_n[x+1] += gain_right;
        if (gain_left != 0) new_n[x-1] += gain_left;
        if (gains[2] != 0) new_n1[x+1] += gains[2];
        if (gains[5] != 0) new_n1[x-1] += gains[5];
    }
    for (auto [x,num] : n1)
    {
        double p_b_to_b_right  = u.y(T-t-1, X-x-1) / u.y(T-t, X-x) * ((1.+u.lambda())/2. - u.lambda() * (u(T-t-1, X-x-1) + u.y(T-t-1, X-x-1)));
        double p_b_to_2b_right = u.y(T-t-1, X-x-1) * u.y(T-t-1, X-x-1) / u.y(T-t, X-x) * u.lambda()/2.;
        double p_b_to_b_left   = u.y(T-t-1, X-x+1) / u.y(T-t, X-x) * ((1.+u.lambda())/2. - u.lambda() * (u(T-t-1, X-x+1) + u.y(T-t-1, X-x+1)));
        double p_b_to_2b_left  = u.y(T-t-1, X-x+1) * u.y(T-t-1, X-x+1) / u.y(T-t, X-x) * u.lambda()/2.;

        // double p_b_to_b_right  = (1.-u.lambda())/2.;
        // double p_b_to_2b_right = u.lambda()/2.;
        // double p_b_to_b_left   = (1.-u.lambda())/2.;
        // double p_b_to_2b_left  = u.lambda()/2.;

        auto gains = multinomial_distribution<ptcl_n>(num, { p_b_to_b_right,
                                                             p_b_to_2b_right,
                                                             p_b_to_b_left,
                                                             p_b_to_2b_left }, det_thr)(engine);
        ptcl_n gain_right = gains[0] + 2*gains[1];
        ptcl_n gain_left = gains[2] + 2*gains[3];
        if (gain_right != 0) new_n1[x+1] += gain_right;
        if (gain_left != 0) new_n1[x-1] += gain_left;
    }
    n = std::move(new_n);
    n1 = std::move(new_n1);
}

void branching_random_walk::evolve(long n_steps, unsigned long det_thr)
{
    for (unsigned i = 0; i != n_steps; ++i)
    {
        evolve_one_step(det_thr);
        ++t;
    }
}

///////////////////////////////////////////////// model /////////////////////////////////////////////////

void model::compute_velocity()
{
    if (lambda_pm == 0)
    {
        velocity = 0;
        return;
    }
    else if (lambda_pm == 1000)
    {
        velocity = 1;
        return;
    }
    else if (lambda_pm > 1000) throw std::invalid_argument("lambda_pm has to be an integer in the interval [0, 1000].");
    real_t gamma1 = sqrt(2*log(1+_lambda));
    std::function<real_t(real_t)> f = [this](real_t g){ return g*std::tanh(g)-std::log((1.+_lambda)*std::cosh(g)); };
    std::function<real_t(real_t)> f_prime = [this](real_t g){ return g*(1-std::pow(std::tanh(g),2)); };
    unsigned int i = 0;
    do
    {
        gamma0 = gamma1;
        gamma1 = gamma0 - f(gamma0) / f_prime(gamma0);
        ++i;
    }
    while (std::abs(gamma1 - gamma0)/gamma0 > 1e-14 && i != 100);
    velocity = std::log((1.+_lambda)*std::cosh(gamma1)) / gamma1;
    gamma0 = gamma1;
    if (i == 100) throw std::domain_error("Newton's method did not reach the required degree of precision within 100 steps.");
}

model::model(long T, long X, unsigned saving_period, double tol) : _X(X), _T(T), _I((T+X)/2), _J((T-X)/2), pm_r(), pm_u(),
            _lambda(lambda_pm / 1000.), logw_minus(std::log((1.-_lambda)/2.)), logw_plus(std::log((1.+_lambda)/2.)), approx_tol(tol), saving_period(saving_period)
{
    output_period = _I > 100 ? _I / 100 : 1;
    compute_velocity();
}

model::model(long T, unsigned saving_period, double tol) : model(T, 0, saving_period, tol)
{
    long bareX = T*velocity - 3./2./gamma0*std::log(T);
    _X = bareX%2 ? bareX + 1 : bareX;
    _I = (T+_X)/2;
    _J = (T-_X)/2;
    output_period = _I > 100 ? _I / 100 : 1;
}

#ifdef PRINT_W_MEMORY
void model::print_w_map()
{
    printf("-\n");
    for (idx i = 0; i != _I + 1; ++i)
    {
        for (idx j = 0; j != _J + 1; ++j)
        {
            if (w_map.count({i,j}) != 0) printf("*");
            else printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}
#endif

long model::u_size()
{
    if (u_map.size() == 0 || u_map.rbegin()->second.size() == 0) return 0;
    else return (sizeof(idx) + sizeof(real_t) * u_map.rbegin()->second.size()) * u_map.size();
}

long model::r_size()
{
    if (logr_map.size() == 0 || logr_map.rbegin()->second.size() == 0) return 0;
    else return (sizeof(idx) + sizeof(real_t) * logr_map.rbegin()->second.size()) * logr_map.size();
}

void model::fill_u_row(idx i) //fill the ith row assuming that the (i-1)th one has already been filled (if i > 0).
{
    if (u_map.count(i) != 0) return; //throw std::runtime_error("The i-th row is already in memory! i = "+i);
    else if (i == 0)
    {
        if (lower_approx_bound.size() == 0) lower_approx_bound.push_back(0);
        if (upper_approx_bound.size() == 0) upper_approx_bound.push_back(0);
        if (u_map.count(i) == 0) u_map[0] = {1.};
    }
    else if (u_map.count(i-1) != 1) throw std::runtime_error("The (i-1)-th row has not been computed yet! i = " + i);
    else
    {
        real_t new_u;
        real_t ui1_j;
        real_t ui_j1 = 0.;
        bool no_approx = (approx_tol == 0.);
        idx upper_loop_limit = no_approx ? _J : 2*_J;
        for (idx j = lower_approx_bound[i-1]; j != upper_loop_limit + 1; ++j)
        {
            ui1_j = u(i-1, j);
            new_u = (1. + _lambda) / 2. * (ui_j1 + ui1_j) - _lambda / 2. * (ui_j1 * ui_j1 + ui1_j * ui1_j);
            ui_j1 = new_u; //the future previous value is the present new value
            if (lower_approx_bound.size() == i && new_u > std::exp(-2 * gamma0 * std::sqrt((double) _T)))
            {
                lower_approx_bound.push_back(j); //the second condition assures that it is only set once.
            }
            if (1. - new_u <= approx_tol && upper_approx_bound.size() == i) //set the upper bound once 
            {
                upper_approx_bound.push_back(j != 0 ? j - 1 : 0);
                return; //Once the upper approximation bound is set, the row is filled.
            }
            if (lower_approx_bound.size() > i) u_map[i].push_back(new_u); //in case new_u is inside the exact computational region
        }
        if (no_approx) upper_approx_bound.push_back(_J);
        else throw std::runtime_error("upper_approximation bound was not set for i = " + i);
    }
}

void model::fill_u()
{
    for (idx i = 0; i <= _I; ++i)
    {
        fill_u_row(i);
        if (i%saving_period != 0 && i > saving_period) u_map.erase(i - saving_period);
        if ( (i + 1) % output_period == 0)
        {
            pm_u.add_datapoint(i / (double) _I);
            std::cout << "\x1B[2J\x1B[H" << "Filling u for lambda = " << d_to_str(_lambda, 3) << " for T = " << _T << "\n" << pm_u << std::endl;
            std::cout << "Memory usage of u: " << u_size() << " byte\n"
                << "Memory usage of r: " << r_size() << " byte" << std::endl;
        }
    }
}

model::real_t model::u(idx i, idx j)
{
    if (i == 0 || upper_approx_bound.size() > i && j > upper_approx_bound.at(i)) return 1.;
    else if (lower_approx_bound.size() > i && j < lower_approx_bound.at(i)) return 0.;
    else
    {
        auto it = u_map.find(i);
        if (it != u_map.end()) return it->second.at(j - lower_approx_bound[i]);
        auto ui1_j = u(i-1,j);
        if (j != 0)
        {
            auto ui_j1 = u(i,j-1);
            return (1. + _lambda) / 2. * (ui_j1 + ui1_j) - _lambda / 2. * (ui_j1*ui_j1 + ui1_j*ui1_j);
        }
        else
        {
            return (1. + _lambda) / 2. * ui1_j - _lambda / 2. * ui1_j * ui1_j;
        }
    }
}

model::real_t model::w(idx i, idx j)
{
    return (1. + _lambda) / 2. - _lambda * u(i,j);
}

model::real_t model::logw(idx i, idx j)
{
    return std::log(w(i,j));
}

model::real_t model::logq(idx i, idx j)
{
    bool checkpoint = !(i%saving_period); //if i is a multiple of saving_period, its congruence class is 0=false, so checkpoint=true
    // if (checkpoint && i > saving_period) model::logq(i-saving_period, j); //if checkpoint, make sure that previous checkpoint gets evaluated first
    // else if (i > saving_period) model::logq(i-i%saving_period, j); //if not checkpoint, make sure that previous checkpoint gets evaluated first
    if (i != 0)
    {
        auto it = logq_map.find({i,j});
        if (it != logq_map.end()) return it->second;
        else
        {
            if (!checkpoint) //if this i-value is not a checkpoint, clean up the values at i-saving_period and i+saving_period
            {
                logq_map.erase({i-saving_period, j});
                logq_map.erase({i+saving_period, j});
            }
            if (j != 0) return logq_map[{i,j}] = logq(i,j-1) + std::log(w(i,j-1) + std::exp(logq(i-1,j)-logq(i,j-1))*w(i-1,j));
            else return logq_map[{i,j}] = logq(i-1,j) + logw(i-1,j);
        }
    }
    else if (j != 0)
    {
        if (!checkpoint) //if this i-value is not a checkpoint, clean up the values at i-saving_period and i+saving_period
        {
            logq_map.erase({i-saving_period, j});
            logq_map.erase({i+saving_period, j});
        }
        return logq_map[{i,j}] = logq(i,j-1) + logw(i,j-1);
    }
    else return 0.;
}

void model::fill_logr_row(idx i) //fill the ith row assuming that the (i+1)th one has already been filled (if i < I).
{
    if (logr_map.count(i) != 0) throw std::runtime_error("The i-th row is already in memory! i = " + i);
    else if (i == _I)
    {
        logr_map[_I].push_front(0.);
        for (idx j = _J-1; j != ((idx) -1); --j)
        {
            logr_map[i].push_front(logr_map.at(_I).front() + logw(i,j));
        }
    }
    else if (logr_map.count(i+1) != 1) throw std::runtime_error("The (i+1)-th row has not been computed yet! i = " + i);
    else
    {
        logr_map[i].push_front(logr_map.at(i+1).at(_J) + logw(i,_J));
        for (idx j = _J-1; j != ((idx) -1); --j)
        {
            real_t logri1_j = logr_map.at(i+1).at(j);
            logr_map[i].push_front(logri1_j + std::log(w(i,j)*(1. + std::exp(logr_map.at(i).front()-logri1_j))));
        }
    }
}

void model::fill_logr()
{
    for (idx i = _I; i != ((idx) -1); --i) //when i == 0, the next subtraction will make it i=max_unsigned - saving_period > I
    {
        fill_logr_row(i);
        if ((i + 1) % output_period == 0)
        {
            pm_r.add_datapoint((_I - (double) i) / _I);
            std::cout << "\x1B[2J\x1B[H" << "Filling r for lambda = " << d_to_str(_lambda, 3) << " for T = " << _T << "\n" << pm_r << std::endl;
            std::cout << "Memory usage of u: " << u_size() << " byte\n"
                << "Memory usage of r: " << r_size() << " byte" << std::endl;
        }
        else if (i > 50) //in this case i+1 is no checkpoint and can be deleted
        {
            logr_map.erase(i+1);
        }
    }
}

model::real_t model::logr(idx i, idx j)
{
    if (i != _I)
    {
        auto it = logr_map.find(i);
        if (it != logr_map.end()) return it->second.at(j);
        else
        {
            auto logri1_j = logr(i+1,j);
            if (j != _J) return logri1_j + std::log(w(i,j)*(1. + std::exp(logr(i,j+1)-logri1_j)));
            else return logri1_j + logw(i,j);
        }
    }
    else if (j != _J)
    {
        return logr(i,j+1) + logw(i,j);
    }
    else return 0.;
}

model::real_t model::logp(idx i, idx j)
{
    return logr(i, j) + logq(i,j);
}

model::real_t model::pplus(idx i, idx j)
{
    if (i == _I) return 0;
    return logw(i,j) + logr(i+1,j) - logr(i,j);
}

void model::print(std::function<double(idx,idx)> field, std::ostream& out, const unsigned& digits, long window_size)
{
    for (int t = 0; t != std::min(_T+1, window_size); ++t)
    {
        for (int x = _I < window_size ? _I : window_size; x != std::max((_X-_T)/2-1, -window_size); --x)
        {
            int i = (t+x)/2;
            int j = (t-x)/2;
            if ((t-x)%2 != 0 || i < 0 || j < 0 || i > _I || j > _J) out << std::setw(digits + 3) << "*  ";
            else out << std::setw(digits + 3) << d_to_str(field(i,j), digits);
        }
        out << "\n" << std::endl;
    }
}

// std::ostream& operator<<(std::ostream& out, const model& M)
// {
//     for (const auto& uij : M.u_map)
//     {
//         out << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10) << uij.first.first << " " << uij.first.second << " " << uij.second << std::endl;
//     }
//     // out << "q" << std::endl;
//     // for (const auto& logqij : M.logq_map)
//     // {
//     //     out << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10) << logqij.first.first << " " << logqij.first.second << " " << logqij.second << std::endl;
//     // }
//     return out;
// }

// std::istream& operator>>(std::istream& in, const model& M)
// {
//     model::idx i;
//     model::idx j;
//     model::real_t uij;
//     while (((in >> std::ws).peek() != 'q') && in >> i && in >> j && in >> uij)
//     {
//         M.u_map[{i,j}] = uij;
//     }
//     // in.ignore();
//     // while (in >> i && in >> j && in >> wqij)
//     // {
//     //     M.logq_map[{i,j}] = wqij;
//     // }
//     // in.clear();
//     return in;
// }