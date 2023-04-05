#include "brw.hpp"
#include <iomanip>
#include <iostream>
#include <cassert>
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

std::string d_to_str(double x, int precision = 2)
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
}

unsigned progress_monitor::time_remaining() const
{
    auto p_it = progress.begin();
    auto t_it = times.begin();
    double sum_tp = 0;
    double sum_tt = 0;
    double sum_t = 0;
    double sum_p = 0;
    unsigned n = times.size();
    if (n < 2) return -1;
    while (p_it != progress.end() && t_it != times.end())
    {
        sum_tp += *t_it * *p_it;
        sum_tt += *t_it * *t_it;
        sum_t += *t_it;
        sum_p += *p_it;
        ++p_it;
        ++t_it;
    }
    double a = (sum_tp - sum_t * sum_p / n) / (sum_tt - sum_t * sum_t / n);
    double b = (sum_p - a * sum_t) / n;
    double temp = (1. - b) / a - times.back();
    return temp;
}

std::ostream& operator<<(std::ostream& out, const progress_monitor& pm)
{
    auto time_left = pm.time_remaining() / 1000; //in seconds
    out << (int) (pm.progress.back()*100) << "%. Remaining time: " << time_left / 3600 << " h " << (time_left % 3600) / 60 << " min " << time_left%60 << " s.";
    return out;
}

void model::compute_velocity()
{
    assert(lambda_pm <= 1000);
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
    double gamma1 = sqrt(2*log(1+_lambda));
    double gamma0;
    std::function<double(double)> f = [this](double g){ return g*std::tanh(g)-std::log((1.+_lambda)*std::cosh(g)); };
    std::function<double(double)> f_prime = [this](double g){ return g*(1-std::pow(std::tanh(g),2)); };
    unsigned int i = 0;
    do
    {
        gamma0 = gamma1;
        gamma1 = gamma0 - f(gamma0) / f_prime(gamma0);
        ++i;
    }
    while (std::abs(gamma1 - gamma0)/gamma0 > 1e-14 && i != 100);
    velocity = std::log((1.+_lambda)*std::cosh(gamma1)) / gamma1;
    if (i == 100) std::cerr << "Newton's method did not achieve the required accuracy for v(gamma0)!" << std::endl;
}

model::model(unsigned lambda_permille, long T, long X, bool approx_w) : _X(X), _T(T), _I((T+X)/2), _J((T-X)/2), approximate_w(approx_w), pm_r(), pm_w()
{
    lambda_pm = lambda_permille;
    _lambda = lambda_pm / 1000.;
    logw_plus = std::log((1.+_lambda)/2.);
    logw_minus = std::log((1.-_lambda)/2.);
    radius_no_approx = _T > 100 ? 3*std::sqrt(_T) : _T;
    saving_period = std::sqrt(_I);

    compute_velocity();
}

model::model(unsigned lambda_permille, long T, bool approx_w) : model(lambda_permille, T, 0, approx_w)
{
    long bareX = T*velocity;
    _X = bareX%2 ? bareX + 1 : bareX;
    _I = (T+_X)/2;
    _J = (T-_X)/2;
    saving_period = std::sqrt(_I);
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

double model::w(idx i, idx j)
{
    #ifdef PRINT_W_MEMORY
    if (w_changed)
    {
        print_w_map();
        w_changed = false;
    }
    #endif
    bool checkpoint = !(i%saving_period); //if i is a multiple of saving_period, its congruence class is 0=false, so checkpoint=true
    // if (checkpoint && i > saving_period) model::w(i-saving_period, j); //if checkpoint, make sure that previous checkpoint gets evaluated first
    // else if (i > saving_period) model::w(i-i%saving_period, j); //if not checkpoint, make sure that previous checkpoint gets evaluated first
    double xdeviation = (1.+velocity)*i+(velocity-1.)*j;
    if (i != 0 && (!approximate_w || std::abs(xdeviation) < radius_no_approx))
    {
        auto it = w_map.find({i,j});
        if (it != w_map.end()) return it->second;
        auto wi1_j = w(i-1,j);
        if (j != 0)
        {
            auto wi_j1 = w(i,j-1);
            if (!checkpoint) //if this i-value is not a checkpoint, clean up the values at i-saving_period and i+saving_period
            {
                w_map.erase({i-saving_period, j});
                w_map.erase({i+saving_period, j});
            }
            else if (-1.1 < xdeviation && xdeviation < 1.1)
            {
                pm_w.add_datapoint(i / (double) _I);
                std::cout << pm_w << std::endl;
            }
            #ifdef PRINT_W_MEMORY
            w_changed = true;
            #endif
            return w_map[{i,j}] = (1.-_lambda*_lambda)/4. + 0.5*(wi_j1*wi_j1 + wi1_j*wi1_j);
        }
        else
        {
            if (!checkpoint) //if this i-value is not a checkpoint, clean up the values at i-saving_period and i+saving_period
            {
                w_map.erase({i-saving_period, j});
                w_map.erase({i+saving_period, j});
            }
            #ifdef PRINT_W_MEMORY
            w_changed = true;
            #endif
            return w_map[{i,j}] = 3./8.-1./8.*_lambda*_lambda+1./4.*_lambda + 0.5*wi1_j*wi1_j;
        }
    }
    else if (xdeviation > radius_no_approx) return (1. + _lambda) / 2.;
    else return (1. - _lambda) / 2.;
}

double model::logw(idx i, idx j)
{
    if (approximate_w)
    {
        double xdeviation = (1.+velocity)*i+(velocity-1.)*j;
        if (xdeviation > radius_no_approx) return logw_plus;
        else if (xdeviation < -radius_no_approx) return logw_minus;
    }
    return std::log(w(i,j));
}

double model::logq(idx i, idx j)
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
    else return 0;
}

double model::logr(idx i, idx j)
{
    bool checkpoint = !(i%saving_period); //if i is a multiple of saving_period, its congruence class is 0=false, so checkpoint=true
    // if (checkpoint && i + saving_period <= _I) model::logr(i+saving_period, j); //if checkpoint, make sure that previous checkpoint gets evaluated first
    // else if (i + saving_period <= _I) model::logr(i + saving_period-i%saving_period, j); //if not checkpoint, make sure that previous checkpoint gets evaluated first
    double xdeviation = (1.+velocity)*i+(velocity-1.)*j;
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
            else if (-1.1 < xdeviation && xdeviation < 1.1)
            {
                pm_r.add_datapoint((_I - (double) i) / _I);
                std::cout << pm_r << std::endl;
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

double model::logp(idx i, idx j)
{
    return logr(i, j) + logq(i,j);
}

double model::pplus(idx i, idx j)
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

std::ostream& operator<<(std::ostream& out, const model& M)
{
    for (const auto& wij : M.w_map)
    {
        out << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10) << wij.first.first << " " << wij.first.second << " " << wij.second << std::endl;
    }
    // out << "q" << std::endl;
    // for (const auto& logqij : M.logq_map)
    // {
    //     out << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10) << logqij.first.first << " " << logqij.first.second << " " << logqij.second << std::endl;
    // }
    return out;
}

std::istream& operator>>(std::istream& in, const model& M)
{
    model::idx i;
    model::idx j;
    double wqij;
    while (((in >> std::ws).peek() != 'q') && in >> i && in >> j && in >> wqij)
    {
        M.w_map[{i,j}] = wqij;
    }
    // in.ignore();
    // while (in >> i && in >> j && in >> wqij)
    // {
    //     M.logq_map[{i,j}] = wqij;
    // }
    // in.clear();
    return in;
}