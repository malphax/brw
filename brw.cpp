#include "brw.hpp"
#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <limits>

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

///////////////////////////////////////////////// u_field /////////////////////////////////////////////////

u_field::u_field(unsigned lambda_pµ, unsigned saving_period) : _lambda_pµ(lambda_pµ), _lambda(lambda_pµ / 1e6), _saving_period(saving_period)
{
    compute_velocity();
    if (lambda_pµ > 1000000) throw std::range_error("Out of range: lambda_pµ must be within [1,1e6].");
}

u_field::real_t u_field::operator()(sidx x, idx t) const
{
    sidx st = t;
    if (t > std::numeric_limits<sidx>::max()) throw std::domain_error("t is to large to be converted to a signed number.");
    if (x < -st) return 1.;
    else if (x > st) return 0.;
    return u(t, (t + x + 1)/2); //if 2|(t+x), the +1 is rounded away, otherwise u(x,t)=u(x+1,t)
}

void u_field::fill_checkpoints(idx T, double tol)
{
    progress_monitor pm;
    idx output_period = T > 100 ? T / 100 : 1;

    idx t_max = T - T%_saving_period;
    idx t_min = (u_map.size() == 0) ? 0 : u_map.rbegin()->first + 1;
    double tol0 = std::exp(-2*_gamma0*std::sqrt(T));
    for (idx t = t_min; t != t_max + 1; ++t)
    {
        fill_row(t, tol0, tol, true);
        if ((t - 1) % _saving_period != 0 && t > _saving_period) u_map.erase(t - 1);
        if ((t + 1) % output_period == 0)
        {
            pm.add_datapoint(t / (double) T);
            std::cout << "\x1B[2J\x1B[H" << "Filling u for lambda = " << d_to_str(_lambda, 6) << " for T = " << T << "\n" << pm << std::endl;
            std::cout << "Memory usage of u: " << estimate_memory() << " byte" << std::endl;
        }
    }
}

void u_field::fill_between(idx T1, idx T2, double tol, bool overwrite)
{
    double tol0 = std::exp(-2*_gamma0*std::sqrt(T2));
    for (idx t = T1; t != T2 + 1; ++t) fill_row(t, tol0, tol, overwrite);
}

std::ostream& operator<<(std::ostream& out, const u_field& u)
{
    out << u._lambda_pµ << " " << u._saving_period << std::endl;
    for (auto& l : u.lower_approx_bound) out << l << " ";
    out << std::endl;
    for (auto& up : u.upper_approx_bound) out << up << " ";
    out << std::endl;
    for (auto& key_value: u.u_map)
    {
        out << key_value.first << " ";
        for (auto& val : key_value.second)
        {
            out << val << " ";
        }
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
    if (!(std::getline(in, line))) throw std::runtime_error("Couldn't read in lower_approx_bound.");
    std::istringstream iss(line);
    u_field::idx i;
    u.lower_approx_bound.clear();
    while (iss >> i) u.lower_approx_bound.push_back(i);
    
    if (!(std::getline(in, line))) throw std::runtime_error("Couldn't read in upper_approx_bound.");
    iss.str(line);
    iss.clear();
    u.upper_approx_bound.clear();
    while (iss >> i) u.upper_approx_bound.push_back(i);

    if(!std::getline(in, line)) throw std::runtime_error("Now values of u to read in.");
    do
    {
        iss.str(line);
        iss.clear();
        u_field::idx t;
        iss >> t;
        u_field::real_t value;
        while (iss >> value) u.u_map[t].push_back(value);
    }
    while (std::getline(in, line));

    return in;
}

unsigned long u_field::estimate_memory(bool rough) const
{
    if (u_map.size() == 0 || u_map.rbegin()->second.size() == 0) return 0;
    else if (rough) return (sizeof(idx) + sizeof(real_t) * u_map.rbegin()->second.size()) * u_map.size();
    unsigned long num_vec_entries = 0;
    for (const auto& kd : u_map) num_vec_entries += kd.second.size();
    return sizeof(idx) * u_map.size() + num_vec_entries;
}

void u_field::print(std::ostream& out, unsigned digits, idx window_size)
{
    for (sidx t = 0; t != window_size; ++t)
    {
        for (sidx x = - (sidx) window_size; x != window_size + 1; ++x)
        {
            int i = (t+x)/2;
            out << std::setw(digits + 3) << d_to_str(u(t,i), digits);
        }
        out << "\n" << std::endl;
    }
}

void u_field::compute_velocity()
{
    if (_lambda_pµ == 0)
    {
        _velocity = 0;
        return;
    }
    else if (_lambda_pµ == 1000000U)
    {
        _velocity = 1;
        return;
    }
    real_t gamma1 = sqrt(2.*log(1.+_lambda));
    std::function<real_t(real_t)> f = [this](real_t g){ return g*std::tanh(g)-std::log((1.+_lambda)*std::cosh(g)); };
    std::function<real_t(real_t)> f_prime = [this](real_t g){ return g*(1-std::pow(std::tanh(g),2)); };
    unsigned int i = 0;
    do
    {
        _gamma0 = gamma1;
        gamma1 = _gamma0 - f(_gamma0) / f_prime(_gamma0);
        ++i;
    }
    while (std::abs(gamma1 - _gamma0)/_gamma0 > 1e-14 && i != 100);
    _velocity = std::log((1.+_lambda)*std::cosh(gamma1)) / gamma1;
    _gamma0 = gamma1;
    if (i == 100) throw std::domain_error("Newton's method did not reach the required degree of precision within 100 steps.");
}

void u_field::fill_row(idx t, double tol0, double tol1, bool overwrite)
{
    if (!overwrite && u_map.count(t) != 0) return;
    else if (t == 0)
    {
        if (lower_approx_bound.size() == 0) lower_approx_bound.push_back(0);
        if (upper_approx_bound.size() == 0) upper_approx_bound.push_back(0);
        if (u_map.count(t) == 0) u_map[0] = {1.};
    }
    else if (u_map.count(t-1) != 1) throw std::runtime_error("The (t-1)-th row has not been computed yet! t = " + std::to_string(t));
    else
    {
        real_t new_u;
        real_t ut1_i;
        real_t ut1_i1 = 1.;
        for (idx i = lower_approx_bound[t-1]; i != t + 1; ++i)
        {
            ut1_i = u(t-1, i);
            new_u = (1. + _lambda) / 2. * (ut1_i1 + ut1_i) - _lambda / 2. * (ut1_i1 * ut1_i1 + ut1_i * ut1_i);
            ut1_i1 = ut1_i; //the future previous value is the present new value
            if (lower_approx_bound.size() == t && 1. - new_u > tol1)
            {
                lower_approx_bound.push_back(i); //The second condition assures that it is only set once.
            }
            if (upper_approx_bound.size() == t && new_u <= tol0) //set the upper bound once 
            {
                upper_approx_bound.push_back(i != 0 ? i - 1 : 0);
                return; //Once the upper approximation bound is set, the row is filled.
            }
            if (lower_approx_bound.size() > t) u_map[t].push_back(new_u); //save u_new iff the lower bound of the exact region has been set.
        }
        //If the program reaches this point, the upper_approximation bound has not been set yet and must be set to its maximum value of t.
        if (upper_approx_bound.size() < lower_approx_bound.size()) upper_approx_bound.push_back(t);
    }
}

u_field::real_t u_field::u(idx t, idx i) const
{
    if (t == 0 && i == 0 || lower_approx_bound.size() > t && i < lower_approx_bound[t]) return 1.;
    else if (upper_approx_bound.size() > t && i > upper_approx_bound[t]) return 0.;
    else
    {
        auto it = u_map.find(t);
        if (it != u_map.end()) return it->second.at(i - lower_approx_bound[t]);
        auto ut1_i = u(t-1,i);
        if (i != 0)
        {
            auto ut1_i1 = u(t-1,i-1);
            return (1. + _lambda) / 2. * (ut1_i1 + ut1_i) - _lambda / 2. * (ut1_i1 * ut1_i1 + ut1_i * ut1_i);
        }
        else return 1.;
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