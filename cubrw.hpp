#ifndef cuBRW_H
#define cuBRW_H

#include <map>
#include <random>
#include <vector>
#include <deque>
#include <numeric>
#include <functional>
#include <chrono>
#include <fstream>
#include <initializer_list>

#define GPU_SUPPORT

#ifdef GPU_SUPPORT
    #include "thrust/device_vector.h"
    #include "thrust/host_vector.h"
    #define DEV_VEC thrust::device_vector<real_t>
    #define HOST_VEC thrust::host_vector<real_t>
    #define COPY thrust::copy
    #define REDUCE thrust::reduce
    #define SWAP thrust::swap
    #define PLUS thrust::plus<real_t>
#else
    #define DEV_VEC std::vector<real_t>
    #define HOST_VEC std::vector<real_t>
    #define COPY std::copy
    #define REDUCE std::reduce
    #define SWAP std::swap
    #define PLUS std::plus<real_t>
#endif

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

class image
{
    private:
        unsigned w, h;
        std::vector<unsigned char> pixels;
    public:
        using rgb = std::array<unsigned char, 3>;
        //Creates a ppm image of format P6 with values 0-255 for each color rgb
        image(unsigned width, unsigned height, unsigned char background_brightness = 255) : w(width), h(height), pixels(3*width*height, background_brightness) {}
        //Saves the file in a binary format
        void save(const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            file << "P6\n" << w << " " << h << "\n255\n";
            file.write(reinterpret_cast<const char *>(&pixels[0]), pixels.size());
        }
        //Read the color values of a given pixel
        rgb pixel(unsigned row, unsigned col) const {
            std::size_t start = 3*row*w + 3*col;
            return { pixels.at(start), pixels.at(start+1), pixels.at(start+2) };
        }
        //Set the color values of a given pixel
        void pixel(unsigned row, unsigned col, rgb colors) {
            std::size_t start = 3*row*w + 3*col;
            pixels.at(start) = std::get<0>(colors);
            pixels.at(start+1) = std::get<1>(colors);
            pixels.at(start+2) = std::get<2>(colors);
        }
        unsigned width() const { return w; }
        unsigned height() const { return h; }
};

class u_field
{
    public:
        using sidx = int; //signed index type
        using idx = unsigned int;
        using real_t = double;
        //This class holds the values of u in the scaling region for a given value of lambda. The saving_period is the distance between two checkpoints.
        u_field(unsigned lambda_pµ, unsigned saving_period = 1);
        /*This class holds the values of u in the scaling region for a given value of lambda. The saving_period is the distance between two checkpoints.
        Using this constructor one also has to pass the x-values of the sites at which the function y(0,x) is non-zero.*/
        template<class It>
        u_field(unsigned lambda_pµ, It x_y0_begin, It x_y0_end, unsigned saving_period = 1) : u_field(lambda_pµ, saving_period)
        {
            _compute_y = true;
            if (x_y0_end-x_y0_begin == 1) throw std::runtime_error("The initializer_list<int> x_y0 must be non-empty if it is provided.");
            int xmin = *std::min_element(x_y0_begin, x_y0_end);
            int xmax = *std::max_element(x_y0_begin, x_y0_end);
            if (xmin <= 0) throw std::domain_error("The sites x at which y(0,x) != 0 must all be strictly positive.");
            for (auto it = x_y0_begin; it != x_y0_end; ++it)
            {
                if (*it % 2 == 0)
                {
                    y0.resize(*it/2 + 1);
                    y0[*it/2] = 1;
                }
            }
        }
        u_field(unsigned lambda_pµ, std::initializer_list<int> x_y0, unsigned saving_period = 1) : u_field(lambda_pµ, x_y0.begin(), x_y0.end(), saving_period) {}
        //Returns u(x,t) assuming step sizes dx == dt == 1
        real_t operator()(idx t, sidx x) const;
        //Returns y(x,t) assuming step sizes dx == dt == 1
        real_t y(idx t, sidx x) const;
        /*Fills in all checkpoints <= T starting from the last row that is saved. If tol = 0., all non-zero values of u are computed.
        device_size is the number of entries that are saved.*/
        void fill_checkpoints(idx T, unsigned device_size, double tol = 1e-5);
        /*
        Fills all rows from inclusively T1 to exclusively T2. Existing rows are overwritten.
        If T1 != 0, row T1-1 must be filled beforehand.
        device_size is the number of entries that are saved.
        */
        void fill_between(idx T1, idx T2, unsigned device_size, double tol = 1e-5);
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
        real_t avg_R (idx t) const { return 2*REDUCE(u_map.at(t).second.crbegin(), u_map.at(t).second.crend(), (real_t) lower_scaling_region_bound(t), PLUS()) - (real_t) t; }
        auto cbegin(idx t) const { return u_map.at(t).second.cbegin(); }
        auto cend(idx t) const { return u_map.at(t).second.cend(); }
        auto cbegin_y(idx t) const { return y_map.at(t).cbegin(); }
        auto cend_y(idx t) const { return y_map.at(t).cend(); }
        //Erases the memory of the rows from inclusively T1 to exclusively T2.
        void erase(idx T1, idx T2) {
            for (unsigned t = T1; t != T2; ++t)
            {
                u_map.erase(t);
                y_map.erase(t);
                prss_map.erase(t);
                plss_map.erase(t);
                prs2s_map.erase(t);
                pls2s_map.erase(t);
            }
        }
        idx lower_scaling_region_bound(idx t) const { return u_map.at(t).first; }
        idx upper_scaling_region_bound(idx t) const { return lower_scaling_region_bound(t) + u_map.at(t).second.size(); }
        void print(std::ostream& out, unsigned digits = 3, idx window_size = 20);
    protected:
        double _lambda;
        unsigned long _lambda_pµ;
        bool _compute_y;
        unsigned _saving_period;
        double _velocity;
        double _gamma0;
        std::map<idx, std::pair<idx, HOST_VEC>> u_map;
        DEV_VEC prev_u;
        DEV_VEC next_u;
        std::map<idx, HOST_VEC> y_map;
        HOST_VEC y0;
        DEV_VEC prev_y;
        DEV_VEC next_y;
        std::map<idx, HOST_VEC> prss_map;
        std::map<idx, HOST_VEC> plss_map;
        std::map<idx, HOST_VEC> prs2s_map;
        std::map<idx, HOST_VEC> pls2s_map;
        idx lsrb;
        void compute_velocity();
        /*Compute row t assuming that row t-1 has been computed (i.p. call illegal for t==0).
        The row is only saved in u_map if save==true. Otherwise it is only temporarily saved until the next call to compute_row.
        Values for which 1-u < tol are not saved.*/
        void compute_row(idx t, double tol = 0., bool save = false);
        //Returns u given (t,i)-coordinates. Only rows that have been filled beforehand can be accessed.
        real_t u_ti(idx t, idx i) const;
        real_t y_ti(idx t, idx i) const;
};

class u_recursion
{
        u_field::real_t lambda;
    public:
        u_recursion(double lambda) : lambda(lambda) {}

        __host__ __device__
        void operator()(u_field::real_t& u, const u_field::real_t& uti, const u_field::real_t& uti1) const {
            u_field::real_t result = (1. + lambda) / 2. * (uti1 + uti) - lambda / 2. * (uti1 * uti1 + uti * uti);
            if (result < 1e-320) u = 0.;
            else u = result;
        };

        __host__ __device__
        void operator()(u_field::real_t& u, const u_field::real_t& uti, const u_field::real_t& uti1,
                        u_field::real_t& y, const u_field::real_t& yti, const u_field::real_t& yti1) const {
            u_field::real_t u_result = (1. + lambda) / 2. * (uti1 + uti) - lambda / 2. * (uti1 * uti1 + uti * uti);
            u_field::real_t y_result = ((1. + lambda) / 2. - lambda * uti) * yti + ((1. + lambda) / 2. - lambda * uti1) * yti1
                                        - lambda / 2. * (yti * yti + yti1 * yti1);
            if (u_result < 1e-320) u = 0.;
            else u = u_result;
            if (y_result < 1e-320) y = 0.;
            else y = y_result;
        };
};

/*class prob_recursion
{
        u_field::real_t lambda;
    public:
        u_recursion(double lambda) : lambda(lambda) {}

        __host__ __device__
        void operator()(u_field::real_t& u, const u_field::real_t& uti, const u_field::real_t& uti1) const {
            u_field::real_t result = (1. + lambda) / 2. * (uti1 + uti) - lambda / 2. * (uti1 * uti1 + uti * uti);
            if (result < 1e-320) u = 0.;
            else u = result;
        };

        __host__ __device__
        void operator()(u_field::real_t& u, const u_field::real_t& uti, const u_field::real_t& uti1,
                        u_field::real_t& y, const u_field::real_t& yti, const u_field::real_t& yti1) const {
            u_field::real_t u_result = (1. + lambda) / 2. * (uti1 + uti) - lambda / 2. * (uti1 * uti1 + uti * uti);
            u_field::real_t y_result = ((1. + lambda) / 2. - lambda * uti) * yti + ((1. + lambda) / 2. - lambda * uti1) * yti1
                                        - lambda / 2. * (yti * yti + yti1 * yti1);
            if (u_result < 1e-320) u = 0.;
            else u = u_result;
            if (y_result < 1e-320) y = 0.;
            else y = y_result;
        };
};*/

template<typename IntType>
class multinomial_distribution
{
    private:
        IntType N;
        std::vector<double> p; //contains the effective probabilities, p_list.size()-1 in number
        IntType threshold;
        bool approximate;
    public:
        /*Defines a multinomial distribution using the probabilities in p_list which need to sum to one.
        The last value is not used but deduced from this assumption.
        */
        multinomial_distribution(IntType n_trials, std::initializer_list<double> p_list) : N(n_trials) {
            p.push_back(*(p_list.begin()));
            double normalization = 1. - p.back();
            for (auto p_it = p_list.begin()+1; p_it != p_list.end()-1; ++p_it)
            {
                p.push_back(*p_it / normalization);
                normalization -= *p_it;
            }
        }
        /*Defines a multinomial distribution using the probabilities in p_list which need to sum to one.
        The last value is not used but deduced from this assumption.
        
        Binomially distributed (n,p) random variables occuring in intermediate steps with n*p > threshold are just taken to be [n*p].
        */
        multinomial_distribution(IntType n_trials, std::initializer_list<double> p_list, IntType threshold) : multinomial_distribution(n_trials, p_list) {
            this->threshold = threshold;
            approximate = true;
        }
        
        static unsigned n_calls_to_binom_dist;

        template<class Generator>
        std::vector<IntType> operator()(Generator& engine);
};

template<typename IntType>
unsigned multinomial_distribution<IntType>::n_calls_to_binom_dist = 0;

template<typename IntType>
template<class Generator>
std::vector<IntType> multinomial_distribution<IntType>::operator()(Generator& engine)
{
    std::vector<IntType> result;
    IntType remaining = N;
    IntType estimated;
    for (auto it = p.begin(); it != p.end(); ++it)
    {
        if (approximate && (estimated = remaining * *it) > threshold) result.push_back(estimated);
        else
        {
            ++n_calls_to_binom_dist;
            result.push_back(std::binomial_distribution<IntType>(remaining, *it)(engine));
        }
        remaining -= result.back();
        if (remaining == 0) break;
    }
    result.push_back(remaining);
    result.resize(p.size()+1);
    return result;
}

class branching_random_walk
{
    protected:
        using ptcl_n = double;
        u_field* ptr_u;
        std::vector<int> red_locations;
        std::map<int, ptcl_n> n_yellow;
        int t;
        int X, T;
        std::mt19937 engine;
        void evolve_one_step(unsigned long det_thr);
    public:
        branching_random_walk(u_field* u_ptr, unsigned random_seed, int X, int T) : ptr_u(u_ptr), engine(random_seed), T(T), X(X), t(0), red_locations{0} { }
        void evolve(long n_steps = 1, unsigned long det_thr = 1<<20);
        auto cbegin() const { return red_locations.cbegin(); }
        auto cend() const { return red_locations.cend(); }
        auto cbegin_y() const { return n_yellow.cbegin(); }
        auto cend_y() const { return n_yellow.cend(); }
        auto size() const { return red_locations.size(); }
        auto size_y() const { return n_yellow.size(); }
        auto y_at(int x) const { if (std::map<int, branching_random_walk::ptcl_n>::const_iterator search = n_yellow.find(x); search != n_yellow.end()) return search->second; else return (branching_random_walk::ptcl_n) 0; }
};

#ifdef GPU_SUPPORT
class red_orange_brw
{
    protected:
        using ptcl_n = double;
        u_field* ptr_u;
        std::vector<int> red_locations;
        thrust::device_vector<ptcl_n> curr_orange;
        thrust::device_vector<ptcl_n> next_orange;
        size_t orange_size;
        int t;
        int X, T, Delta;
        std::mt19937 engine;
        void evolve_one_step(unsigned long det_thr);
    public:
        red_orange_brw(u_field* u_ptr, unsigned random_seed, int X, int T, int Delta) : ptr_u(u_ptr), engine(random_seed), T(T), X(X), t(0), Delta(Delta), red_locations{0} {
            orange_size = u_ptr->cend(0) - u_ptr->cbegin(0);
            curr_orange = thrust::device_vector<ptcl_n>(orange_size, 0);
            next_orange = thrust::device_vector<ptcl_n>(orange_size, 0);
        }
        void evolve(long n_steps = 1, unsigned long det_thr = 1<<20) { for (unsigned i = 0; i != n_steps; ++i) { evolve_one_step(det_thr); ++t; } }
        auto cbegin_r() const { return red_locations.cbegin(); }
        auto cend_r() const { return red_locations.cend(); }
        auto cbegin_o() const { return curr_orange.cbegin(); }
        auto cend_o() const { return curr_orange.cend(); }
        auto size_r() const { return red_locations.size(); }
        auto size_o() const { return orange_size; }
};
#endif

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