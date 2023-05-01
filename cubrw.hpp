#ifndef cuBRW_H
#define cuBRW_H

//#define PRINT_W_MEMORY

#include <map>
#include <random>
#include <vector>
#include <deque>
#include <functional>
#include <chrono>

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

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
        void reset();
        //Estimated remaining time in milliseconds
        unsigned time_remaining() const;
        void add_datapoint(double percent_progress);
        friend std::ostream& operator<<(std::ostream& out, const progress_monitor& pm);
};

class timer
{
    private:
        std::chrono::_V2::steady_clock::time_point starting_time;
    public:
        timer() { reset(); }
        void reset() {starting_time = std::chrono::steady_clock::now(); }
        //Returns passed time since last reset or construction in milliseconds.
        unsigned long long time() const { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-starting_time).count(); }
        friend std::ostream& operator<<(std::ostream& out, const timer& t);
};

class u_field
{
    public:
        using sidx = int; //signed index type
        using idx = unsigned int;
        using real_t = double;
        //This class holds the values of u in the scaling region for a given value of lambda. The saving_period is the distance between two checkpoints.
        u_field(unsigned lambda_pµ, unsigned saving_period = 1);
        //Returns u(x,t) assuming step sizes dx == dt == 1
        real_t operator()(sidx x, idx t) const;
        /*Fills in all checkpoints <= T starting from the last row that is saved. If tol = 0., all non-zero values of u are computed.
        Values for which 1-u < tol or u < exp(-2*gamma0*sqrt(T2)) are not saved.*/
        void fill_checkpoints(idx T, double tol = 0.);
        /*
        Fills all rows from inclusively T1 to inclusively T2.
        If overwrite == true, already filled rows including checkpoints will be deleted first.
        If overwrite == false, rows that are already filled are skipped.
        Values for which 1-u < tol or u < exp(-2*gamma0*sqrt(T2)) are not saved.
        */
        void fill_between(idx T1, idx T2, double tol = 0., bool overwrite = false);
        /*Writes the object to "out". The first row contains lambda and the saving_period.
        The second and third rows contain all values of lower_approx_bound and upper_approx_bound.
        The remaining lines start with the time index and afterwards the entries of u at this time index.
        Only the values that are saved are outputted.*/
        friend std::ostream& operator<<(std::ostream& out, const u_field& u);
        //Reads in the object from "in", following the format outputted by operator<<. this->u_map gets cleared and overwritten.
        friend std::istream& operator>>(std::istream& in, u_field& u);
        //Output the approximate memory used to save u. If rough == true, it is assumed that all rows contain the same number of elements as the latest row.
        unsigned long estimate_memory(bool rough = true) const;
        double lambda() const { return _lambda; }
        unsigned saving_period() const { return _saving_period; }
        double velocity() const { return _velocity; }
        double gamma0() const { return _gamma0; }
        unsigned number_of_threads() const { return _nthreads; }
        void number_of_threads(unsigned n);
        real_t avg_R (idx t) const { return 2*thrust::reduce(u_map.at(t).second.crbegin(), u_map.at(t).second.crend(), (real_t) lower_scaling_region_bound(t), thrust::plus<real_t>()) - (real_t) t; }
        auto cbegin(idx t) const { return u_map.at(t).second.cbegin(); }
        auto cend(idx t) const { return u_map.at(t).second.cend(); }
        idx lower_scaling_region_bound(idx t) const { return u_map.at(t).first; }
        idx upper_scaling_region_bound(idx t) const { return lower_scaling_region_bound(t) + u_map.at(t).second.size(); }
        void print(std::ostream& out, unsigned digits = 3, idx window_size = 20);
    private:
        double _lambda;
        unsigned long _lambda_pµ;
        unsigned _saving_period;
        double _velocity;
        double _gamma0;
        unsigned _nthreads;
        std::map<idx, std::pair<idx,thrust::device_vector<real_t>>> u_map;
        thrust::device_vector<real_t> prev_u;
        thrust::device_vector<real_t> next_u;
        idx lsrb;
        void compute_velocity();
        /*Compute row t assuming that row t-1 has been computed (i.p. call illegal for t==0).
        The row is only saved in u_map if save==true. Otherwise it is only temporarily saved until the next call to compute_row.
        Values for which 1-u < tol are not saved.*/
        void compute_row(idx t, double tol = 0., bool save = false);
        //Returns u given (t,i)-coordinates. Only rows that have been filled beforehand can be accessed.
        real_t u(idx t, idx i) const;
};

class u_recursion
{
        u_field::real_t lambda;
    public:
        u_recursion(double lambda) : lambda(lambda) {}
        __host__ __device__
        u_field::real_t operator()(u_field::real_t ut1i, u_field::real_t ut1i1) const {
            u_field::real_t result = (1. + lambda) / 2. * (ut1i1 + ut1i) - lambda / 2. * (ut1i1 * ut1i1 + ut1i * ut1i);
            if (result < 1e-320) return 0.;
            return result;
        };
};

class model
{
    public:
        using idx = unsigned;
        using real_t = double;
    private:
        const unsigned lambda_pm = 10;
        const double approx_tol;
        const real_t _lambda;
        const real_t logw_plus;
        const real_t logw_minus;
        inline static real_t velocity;
        inline static real_t gamma0;
        unsigned output_period;
        inline static std::map<idx, std::vector<real_t>> u_map;
        inline static std::map<std::pair<idx, idx>, real_t> logq_map;
        std::map<idx, std::deque<real_t>> logr_map;
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
        void fill_logr_row(idx i);
        long u_size();
        long r_size();
    public:
        inline static std::vector<idx> lower_approx_bound;
        inline static std::vector<idx> upper_approx_bound;
        bool approximate_u;
        const unsigned saving_period;
        model(long T, long X, unsigned saving_period, double tol = 0.);
        model(long T, unsigned saving_period, double tol = 0.);
        void fill_u();
        void fill_logr();
        idx m_t(idx i); // gives the index j such that [i,j] corresponds to the coordinates (t,m_t)
        real_t u(idx i, idx j);
        real_t w(idx i, idx j);
        real_t logw(idx i, idx j);
        real_t logq(idx i, idx j);
        real_t logr(idx i, idx j);
        real_t logp(idx i, idx j);
        real_t pplus(idx i, idx j);
        inline const real_t& lambda() const { return _lambda; }
        inline const real_t& v() const { return velocity; }
        inline const real_t& gamma() const { return gamma0; }
        inline const unsigned& out_period() const { return output_period; }
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