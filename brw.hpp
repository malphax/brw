#ifndef BRW_HPP
#define BRW_HPP

//#define PRINT_W_MEMORY

#include <map>
#include <random>
#include <vector>
#include <deque>
#include <functional>
#include <chrono>

class BRW
{
    private:
        typedef unsigned long particle_number;
        double lam;
        std::map<long, particle_number> n;
        std::vector<long> rightmost;
        long time = 0;
        static std::mt19937 engine;
        static const unsigned seed = 5;
        void evolve_one_step(unsigned long det_thr);
    public:
        BRW(double lambda) : lam(lambda) { n[0] = 1; }
        void evolve(long n_steps = 1, unsigned long det_thr = 1<<20);
        inline long t() const { return time; };
        inline double n_at(long x) const;
        inline auto cbegin() const { return n.cbegin(); }
        inline auto cend() const { return n.cend(); }
        inline const std::vector<long>& rightmost_track() const { return rightmost; }
};

inline double BRW::n_at(long x) const
{
    auto nx_it = n.find(x);
    if (nx_it == n.end()) return 0.;
    return nx_it->second;
}

class progress_monitor
{
    private:
        std::chrono::_V2::steady_clock::time_point starting_time;
        std::deque<double> progress; //values between 0 and 1
        std::deque<unsigned long> times; //in milliseconds and counted since start
    public:
        unsigned size;
        progress_monitor(unsigned size = 10) : size(size) { reset(); }
        void reset(); //also resets
        unsigned time_remaining() const; //in milliseconds
        void add_datapoint(double percent_progress);
        friend std::ostream& operator<<(std::ostream& out, const progress_monitor& pm);
};

class model
{
    public:
        using idx = unsigned;
        using real_t = long double;
    private:
        const unsigned lambda_pm = 10;
        const double approx_tol;
        const real_t _lambda;
        const real_t logw_plus;
        const real_t logw_minus;
        inline static real_t velocity;
        inline static real_t gamma0;
        inline static std::map<idx, std::vector<real_t>> u_map;
        inline static std::map<std::pair<idx, idx>, real_t> logq_map;
        std::map<std::pair<idx, idx>, real_t> logr_map;
        void compute_velocity();
        #ifdef PRINT_W_MEMORY
        void print_w_map();
        bool w_changed = false;
        #endif
        long _X, _T;
        idx _I, _J;
        progress_monitor pm_u;
        progress_monitor pm_r;
        void fill_u_row(idx i);
    public:
        inline static std::vector<idx> lower_approx_bound;
        inline static std::vector<idx> upper_approx_bound;
        bool approximate_u;
        const unsigned saving_period;
        model(long T, long X, unsigned saving_period, double tol = 0.);
        model(long T, unsigned saving_period, double tol = 0.);
        void fill_u();
        void fill_logr();
        real_t u(idx i, idx j);
        real_t w(idx i, idx j);
        real_t logw(idx i, idx j);
        real_t logq(idx i, idx j);
        real_t logr(idx i, idx j);
        real_t logp(idx i, idx j);
        real_t pplus(idx i, idx j);
        inline const real_t& lambda() const { return _lambda; }
        inline const real_t& v() const { return velocity; }
        inline long X() const { return _X; }
        inline long T() const { return _T; }
        inline idx I() const { return _I; }
        inline idx J() const { return _J; }
        friend std::ostream& operator<<(std::ostream& out, const model& M);
        friend std::istream& operator>>(std::istream& in, const model& M);
        void print(std::function<double(model::idx, model::idx)> field, std::ostream& out, const unsigned& digits = 3, long window_size = 20);
};

std::string d_to_str(double x, int precision = 2);

template<typename T>
std::pair<T, T> linear_fit(const std::vector<T>& x, const std::vector<T>& y)
{
    auto x_it = x.begin();
    auto y_it = y.begin();
    T sum_xy = 0;
    T sum_xx = 0;
    T sum_x = 0;
    T sum_y = 0;
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
    T k = (sum_xy - sum_x * sum_y / n) / (sum_xx - sum_x * sum_x / n);
    T d = (sum_y - k * sum_x) / n;
    return {k,d};
}

#endif