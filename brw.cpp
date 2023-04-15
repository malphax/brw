#include "brw.hpp"
#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <limits>

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
    compute_velocity();
}

model::model(long T, unsigned saving_period, double tol) : model(T, 0, saving_period, tol)
{
    long bareX = T*velocity - 3./2./gamma0*std::log(T);
    _X = bareX%2 ? bareX + 1 : bareX;
    _I = (T+_X)/2;
    _J = (T-_X)/2;
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

void model::fill_u_row(idx i) //fill the ith row assuming that the (i-1)th one has already been filled (if i > 0).
{
    if (i == 0)
    {
        if (lower_approx_bound.size() == 0) lower_approx_bound.push_back(0);
        if (upper_approx_bound.size() == 0) upper_approx_bound.push_back(0);
        if (u_map.count(i) == 0) u_map[0] = {1.};
    }
    else if (u_map.count(i) != 0) throw std::runtime_error("The i-th row is already in memory! i = "+i);
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
            if (new_u > approx_tol && lower_approx_bound.size() == i) lower_approx_bound.push_back(j); //the second condition assures that it is only set once.
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
    idx output_period = _I > 100 ? _I / 100 : 1;
    for (idx i = 0; i <= _I; ++i)
    {
        fill_u_row(i);
        if (i%saving_period != 0 && i > saving_period) u_map.erase(i - saving_period);
        if ( (i + 1) % output_period == 0)
        {
            pm_u.add_datapoint(i / (double) _I);
            std::cout << "\x1B[2J\x1B[H" << "Filling u for lambda = " << d_to_str(_lambda, 3) << " for T = " << _T << "\n" << pm_u << std::endl;
            std::cout << "Memory usage of u: " << (sizeof(idx) + sizeof(real_t)) * u_map.size() << " byte\n"
                << "Memory usage of r: " << (2 * sizeof(idx) + sizeof(real_t)) * logr_map.size() << " byte" << std::endl;
        }
    }
}

void model::fill_logr()
{
    idx output_period = _I > 100 ? _I / 100 : 1;
    for (unsigned i = _I-_I%saving_period; i <= _I; i -= saving_period) //when i == 0, the next subtraction will make it i=max_unsigned - saving_period > I
    {
        logr(i, 0);
        if ( (i + 1) % output_period == 0)
        {
            pm_r.add_datapoint((_I - (double) i) / _I);
            std::cout << "\x1B[2J\x1B[H" << "Filling r for lambda = " << d_to_str(_lambda, 3) << " for T = " << _T << "\n" << pm_r << std::endl;
            std::cout << "Memory usage of u: " << (sizeof(idx) + sizeof(real_t)) * u_map.size() << " byte\n"
                    << "Memory usage of r: " << (2 * sizeof(idx) + sizeof(real_t)) * logr_map.size() << " byte" << std::endl;
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

model::real_t model::logr(idx i, idx j)
{
    bool checkpoint = !(i%saving_period); //if i is a multiple of saving_period, its congruence class is 0=false, so checkpoint=true
    // if (checkpoint && i + saving_period <= _I) model::logr(i+saving_period, j); //if checkpoint, make sure that previous checkpoint gets evaluated first
    // else if (i + saving_period <= _I) model::logr(i + saving_period-i%saving_period, j); //if not checkpoint, make sure that previous checkpoint gets evaluated first
    real_t xdeviation = (1.+velocity)*i+(velocity-1.)*j;
    if (i != _I)
    {
        auto it = logr_map.find({i,j});
        if (it != logr_map.end()) return it->second;
        else
        {
            if (!checkpoint) //if this i-value is not a checkpoint, clean up the values at i-saving_period and i+saving_period
            {
                logr_map.erase({i-saving_period, j});
                logr_map.erase({i+saving_period, j});
            }
            auto logri1_j = logr(i+1,j);
            if (j != _J) return logr_map[{i,j}] = logri1_j + std::log(w(i,j)*(1. + std::exp(logr(i,j+1)-logri1_j)));
            else return logr_map[{i,j}] = logri1_j + logw(i,j);
        }
    }
    else if (j != _J)
    {
        if (!checkpoint) //if this i-value is not a checkpoint, clean up the values at i-saving_period and i+saving_period
        {
            logr_map.erase({i-saving_period, j});
            logr_map.erase({i+saving_period, j});
        }
        return logr_map[{i,j}] = logr(i,j+1) + std::log(w(i,j));
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