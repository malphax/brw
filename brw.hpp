#ifndef BRW_HPP
#define BRW_HPP

//#define PRINT_W_MEMORY

#include <map>
#include <random>
#include <vector>
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
        std::vector<double> progress; //values between 0 and 1
        std::vector<unsigned long> times; //in milliseconds and counted since start
    public:
        progress_monitor() { reset(); }
        void reset(); //also resets
        unsigned time_remaining() const; //in milliseconds
        void add_datapoint(double percent_progress);
        friend std::ostream& operator<<(std::ostream& out, const progress_monitor& pm);
};

class model
{
    public:
        typedef unsigned long idx;
    private:
        inline static unsigned lambda_pm;
        inline static double _lambda;
        inline static double logw_plus;
        inline static double logw_minus;
        inline static double velocity;
        inline static std::map<std::pair<idx, idx>, double> w_map;
        inline static std::map<std::pair<idx, idx>, double> logq_map;
        std::map<std::pair<idx, idx>, double> logr_map;
        void compute_velocity();
        #ifdef PRINT_W_MEMORY
        void print_w_map();
        bool w_changed = false;
        #endif
        long _X, _T;
        idx _I, _J;
        progress_monitor pm_w;
        progress_monitor pm_r;
    public:
        bool approximate_w;
        long radius_no_approx;
        unsigned long saving_period;
        model(unsigned lambda_permille, long T, long X, bool approx_w = false);
        model(unsigned lambda_permille, long T, bool approx_w = false);
        double w(idx i, idx j);
        double logw(idx i, idx j);
        double logq(idx i, idx j);
        double logr(idx i, idx j);
        double logp(idx i, idx j);
        double pplus(idx i, idx j);
        inline const double& lambda() const { return _lambda; }
        inline const double& v() const { return velocity; }
        inline long X() const { return _X; }
        inline long T() const { return _T; }
        inline idx I() const { return _I; }
        inline idx J() const { return _J; }
        friend std::ostream& operator<<(std::ostream& out, const model& M);
        friend std::istream& operator>>(std::istream& in, const model& M);
        void print(std::function<double(model::idx, model::idx)> field, std::ostream& out, const unsigned& digits = 3, long window_size = 20);
};


#endif