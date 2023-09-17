#ifndef BRW_HPP
#define BRW_HPP

#include <vector>
#include <deque>
#include <random>
#include <initializer_list>
#include <map>
#include <chrono>
#include <fstream>
#include <array>
#include <algorithm>

namespace brw
{
    /////////////////
    ////stopwatch////
    /////////////////
    /*!
     * @brief A simple class for measuring time delays.
     *
     * Use reset() to restart the timer and time() to measure the time since the last start.
     * 
     */
    class stopwatch
    {
    private:
        std::chrono::_V2::steady_clock::time_point starting_time;
    public:
        //! Creates a stopwatch and starts it.
        stopwatch() { reset(); }
        
        //! Restarts the stopwatch.
        void reset() {starting_time = std::chrono::steady_clock::now(); }
        
        //! Returns passed time since last reset or construction in milliseconds.
        unsigned long long time() const { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-starting_time).count(); }
        
        /*!
         * @brief Prints time since last reset in hours, minutes, seconds and milliseconds.
         * 
         * @param out output stream
         * @param s stopwatch to print
         * @return A reference to out.
         */
        friend std::ostream& operator<<(std::ostream& out, const stopwatch& s);
    };

    ////////////////////////
    ////progress_monitor////
    ////////////////////////
    /*!
     * @brief A utility class to keep track of progress.
     * 
     * Allows to save the progress at various times to estimate the remaining time of a progress (using linear regression).
     */
    class progress_monitor
    {
        private:
            std::chrono::_V2::steady_clock::time_point starting_time;
            std::deque<double> progress; //!< values between 0 and 1
            std::deque<unsigned long> times; //!< in milliseconds and counted since starting_time
            std::pair<double, double> linear_fit(const std::vector<double>& x, const std::vector<double>& y) const;
        public:
            unsigned size;
            /*!
             * @brief Construct a new progress monitor object.
             * 
             * @param size Number of data points to save and use for the extrapolation.
             */
            progress_monitor(unsigned size = 10) : size(size) { reset(); }
            //! Deletes all data points saved so far and resets the internal clock. After calling this, \texttt{*this} is in the same state as a freshly constructed \texttt{progress_monitor}
            void reset();
            /*! @brief Estimation of remaining time of the process.
             * 
             * The estimate is based on a linear extrapolation of the data points collected thus far.
             * @return Estimated time remaining in milliseconds.
             */
            unsigned time_remaining() const;
            //! Takes a floating point number in [0,1] indicating the progress of some task. Automatically saves the time relative to the last reset.
            void add_datapoint(double percent_progress);
            /*! Prints the progress and estimated the time in a formatted way to out.
             *
             * @return reference to out
             */
            friend std::ostream& operator<<(std::ostream& out, const progress_monitor& pm);
    };

    ///////////
    ////ppm////
    ///////////
    /*!
     * @brief A class for creating ppm images.
     * 
     */
    class ppm
    {
        private:
            unsigned w, h;
            std::vector<unsigned char> pixels;
        public:
            using rgb = std::array<unsigned char, 3>;
            //! Creates a ppm image of format P6 with values 0-255 for each color rgb.
            ppm(unsigned width, unsigned height, unsigned char background_brightness = 255) : w(width), h(height), pixels(3*width*height, background_brightness) {}
            //! Saves the file in a binary format.
            void save(const std::string& filename) {
                std::ofstream file(filename, std::ios::binary);
                file << "P6\n" << w << " " << h << "\n255\n";
                file.write(reinterpret_cast<const char *>(&pixels[0]), pixels.size());
            }
            /*! @brief Reads the color values of a given pixel.
             * @return RGB color values embedded in type std::array<unsigned char, 3>
             */
            rgb pixel(unsigned row, unsigned col) const {
                std::size_t start = 3*row*w + 3*col;
                return { pixels.at(start), pixels.at(start+1), pixels.at(start+2) };
            }
            /*!
             * @brief Sets the color values of a given pixel.
             * 
             * rgb is an alias for std::array<unsigned char, 3>
             */
            void pixel(unsigned row, unsigned col, rgb colors) {
                std::size_t start = 3*row*w + 3*col;
                pixels.at(start) = std::get<0>(colors);
                pixels.at(start+1) = std::get<1>(colors);
                pixels.at(start+2) = std::get<2>(colors);
            }
            //! Returns the width of the image in pixels.
            unsigned width() const { return w; }
            //! Returns the height of the image in pixels.
            unsigned height() const { return h; }
    };

    ////////////////////////////////
    ////multinomial_distribution////
    ////////////////////////////////
    /*!
     * @brief A class for defining multinomial distributions.
     *
     * The multinomial distribution \f$M(n,p_1,p_2,...,p_m)\f$ corresponds to colouring \f$n\f$ objects in \f$m\f$ colours where each object independently has probability \f$p_i\f$ to be coloured with colour \f$i\in\{1,...,m\}\f$.
     * 
     * @tparam IntType An integer type.
     */
    template<typename IntType>
    class multinomial_distribution
    {
        private:
            IntType N;
            std::vector<double> p; //contains the effective probabilities, p_list.size()-1 in number
            IntType threshold;
            bool approximate;
        public:
            /*! 
             * @brief Constructs a multinomial distribution.
             * 
             * Defines a multinomial distribution using the probabilities in p_list which need to sum to one.
             * The last provided value is not used but deduced from this assumption.
             * @param n_trials Total number \f$n\f$ of objects
             * @param p_list Initializer list of the probabilities \f$p_1,...,p_m\f$
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
            /*!
             * @brief Constructs a multinomial distribution with a threshold.
             * 
             * Defines a multinomial distribution using the probabilities in p_list which need to sum to one.
             * The last provided value is not used but deduced from this assumption.
             *             
             * Binomially distributed \f$\mathcal{B}(n,p)\f$ random variables occuring in intermediate steps with \f$np > \texttt{threshold}\f$ are just taken to be \f$\lfloor np\rfloor\f$.
             * @param n_trials Total number \f$n\f$ of objects
             * @param p_list Initializer list of the probabilities \f$p_1,...,p_m\f$
             * @param threshold see description above
             */
            multinomial_distribution(IntType n_trials, std::initializer_list<double> p_list, IntType threshold) : multinomial_distribution(n_trials, p_list) {
                this->threshold = threshold;
                approximate = true;
            }

            /*!
             * @brief Generates a random vector.
             * 
             * @tparam Generator A random number engine type deduced from engine.
             * @param engine A random number engine.
             * @return std::vector<IntType> A vector \f$\{c_1,...,c_m\}\f$ where \f$c_i\f$ is the number of objects coloured in \f$i\f$.
             */
            template<class Generator>
            std::vector<IntType> operator()(Generator& engine);

            /*!
             * @brief Counter of drawn binomial random variables.
             * 
             * The counter is shared among all instances of this classes.
             */
            static unsigned n_calls_to_binom_dist;
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

    ///////////////////
    ////prob_fields////
    ///////////////////

    //! A class for storing u and y.
    class prob_fields
    {
    public:
        using sidx = int; //! signed index type
        using idx = unsigned int;
        using real_t = double;
        /*!
         * @brief Constructs a prob_fields object.
         * This class holds the values of u in the scaling region for a given value of lambda.
         * @param lambda_pµ \f$\lambda\cdot10^6\f$ where λ is the splitting rate
         * @param saving_period the distance between two checkpoints
         */
        prob_fields(unsigned lambda_pµ, unsigned saving_period = 1);

        /*!
         * @brief Constructs a prob_fields object.
         * This class holds the values of u and y in the scaling region for a given value of lambda.
         * @param x_y0_begin begin iterator to the x-values of the sites at which the function y(0,x) is non-zero
         * @param x_y0_end end iterator to the x-values of the sites at which the function y(0,x) is non-zero
         * @param lambda_pµ \f$\lambda\cdot10^6\f$ where λ is the splitting rate
         * @param saving_period the distance between two checkpoints
         */
        template<class It>
        prob_fields(unsigned lambda_pµ, It x_y0_begin, It x_y0_end, unsigned saving_period = 1) : prob_fields(lambda_pµ, saving_period)
        {
            _compute_y = true;
            if (x_y0_end-x_y0_begin == 1) throw std::runtime_error("The initializer_list<int> x_y0 must be non-empty if it is provided.");
            int xmin = *std::min_element(x_y0_begin, x_y0_end);
            // int xmax = *std::max_element(x_y0_begin, x_y0_end);
            if (xmin <= 0) throw std::domain_error("The sites x at which y(0,x) != 0 must all be strictly positive.");
            for (auto it = x_y0_begin; it != x_y0_end; ++it)
            {
                if (*it % 2 == 0)
                {
                    y_map[0].resize(*it/2 + 1);
                    y_map[0][*it/2] = 1;
                }
            }
        }

        /*!
         * @brief Constructs a prob_fields object.
         * This class holds the values of u and y in the scaling region for a given value of lambda.
         * @param lambda_pµ \f$\lambda\cdot10^6\f$ where λ is the splitting rate
         * @param x_y0 the x-values of the sites at which the function y(0,x) is non-zero
         * @param saving_period the distance between two checkpoints
         */
        prob_fields(unsigned lambda_pµ, std::initializer_list<int> x_y0, unsigned saving_period = 1) : prob_fields(lambda_pµ, x_y0.begin(), x_y0.end(), saving_period) {}
        
        //! Returns u(t,x) assuming step sizes dx == dt == 1
        real_t u(idx t, sidx x) const;

        //! Returns y(t,x) assuming step sizes dx == dt == 1
        real_t y(idx t, sidx x) const;

        /*!
         * @brief Fill in checkpoints up to T.
         * @param T All checkpoints up to and including time T starting from the last row that is saved.
         * @param device_size the number of entries that are saved per timestep
         * @param tol tolerance s.t. values of u greater than \f$1-\mathrm{tol}\f$ are not stored
         */
        void fill_checkpoints(idx T, unsigned device_size, double tol = 1e-5);

        /*!
         * @brief Fill all rows from inclusively T1 to exclusively T2.
         * Existing rows are overwritten.
         * If T1 != 0, row T1-1 must have been filled beforehand.
         * @param T1 first row to fill
         * @param T2 second row to fill
         * @param device_size the number of entries that are saved per timestep
         * @param tol tolerance s.t. values of u greater than \f$1-\mathrm{tol}\f$ are discarded
         */
        void fill_between(idx T1, idx T2, unsigned device_size, double tol = 1e-5);

        //! @brief Output the approximate memory used to save u.
        unsigned long estimate_memory() const;
        double lambda() const { return _lambda; }
        unsigned saving_period() const { return _saving_period; }
        double velocity() const { return _velocity; }
        double gamma0() const { return _gamma0; }

        //! @brief Compute \f$m_t\f$.
        real_t avg_M (idx t) const { return 2*std::accumulate(u_map.at(t).second.crbegin(), u_map.at(t).second.crend(), (real_t) lower_scaling_region_bound(t)) - (real_t) t; }
        auto cbegin_u(idx t) const { return u_map.at(t).second.cbegin(); }
        auto cend_u(idx t) const { return u_map.at(t).second.cend(); }
        auto cbegin_y(idx t) const { return y_map.at(t).cbegin(); }
        auto cend_y(idx t) const { return y_map.at(t).cend(); }
        //! Erase the memory of the rows from inclusively T1 to exclusively T2.
        void erase(idx T1, idx T2) { for (unsigned t = T1; t != T2; ++t) { u_map.erase(t); y_map.erase(t); } }
        idx lower_scaling_region_bound(idx t) const { return u_map.at(t).first; }
        idx upper_scaling_region_bound(idx t) const { return lower_scaling_region_bound(t) + u_map.at(t).second.size(); }
    private:
        double _lambda;
        unsigned long _lambda_pµ;
        bool _compute_y;
        unsigned _saving_period;
        double _velocity;
        double _gamma0;
        std::map<idx, std::pair<idx, std::vector<real_t>>> u_map;
        std::vector<real_t> prev_u;
        std::vector<real_t> next_u;
        idx lsrb;
        std::map<idx, std::vector<real_t>> y_map;
        std::vector<real_t> prev_y;
        std::vector<real_t> next_y;

        //! Compute the front velocity using Newton's method.
        void compute_velocity();

        /*!
         * @brief Compute a value of \f$u\f$ from two previous ones.
         * 
         * @param uti \f$u[t,i]\f$
         * @param uti1 \f$u[t,i-1]\f$
         * @return real_t \f$u[t+1,i]\f$
         */
        real_t u_recursion(real_t uti, real_t uti1) { return (1. + _lambda) / 2. * (uti1 + uti) - _lambda / 2. * (uti1 * uti1 + uti * uti); }

        /*!
         * @brief Compute a value of \f$u\f$ from two previous ones.
         * 
         * @param uti \f$u[t,i]\f$
         * @param uti1 \f$u[t,i-1]\f$
         * @return real_t \f$u[t+1,i]\f$
         */
        real_t y_recursion(real_t uti, real_t uti1, real_t yti, real_t yti1) { return ((1. + _lambda) / 2. - _lambda * uti) * yti + ((1. + _lambda) / 2. - _lambda * uti1) * yti1 - _lambda / 2. * (yti * yti + yti1 * yti1); }

        /*!
         * @brief Compute row t assuming that row t-1 has been computed.
         * @warning call illegal for t==0
         * 
         * The row is only saved in u_map if save==true.
         * Otherwise it is only temporarily saved until the next call to compute_row.
         * Values for which 1-u < tol are not saved.
         */
        void compute_row(idx t, double tol = 0., bool save = false);
        //! Returns u given (t,i)-coordinates. Only rows that have been filled beforehand can be accessed.
        real_t u_ti(idx t, idx i) const;
        //! Returns y given (t,i)-coordinates. Only rows that have been filled beforehand can be accessed.
        real_t y_ti(idx t, idx i) const;
    };

    /*!
     * @brief Class for generating conditioned branching random walks.
     * 
     * @tparam RandEng The type of random engine used for the generation.
     */
    template<class RandEng>
    class condBRW
    {
        private:
            using ptcl_n = double;
            prob_fields* ptr_uy;
            std::vector<int> red_locations; //!< There are very few red particles, thus it is easier to save the position of each one.
            std::vector<ptcl_n> n_yellow; //!< Yellow particle numbers grow exponentially, so we only track the number per site.
            int t;
            int X, T;
            RandEng engine;

            /*!
             * @brief Evolve the state by one timestep.
             * 
             * This function encodes all of the classes dynamics.
             * It determines the rules according to which all particles move and replicate.
             * 
             * @param det_thr During internal computations, \f$\mathcal{B}(n,p)\f$ distributed random variables are replaced by np \f$np>\texttt{det_thr}\f$.
             */
            void evolve_one_step(unsigned long det_thr);
        public:
            /*!
             * @brief Construct a new conditioned branching random walk object.
             * 
             * @param uy_ptr pointer to a prob_fields object
             * @param random_seed the random seed for the generation
             * @param X the destination of the red particle
             * @param T the number of simulation steps
             */
            condBRW(prob_fields* uy_ptr, unsigned random_seed, int X, int T) : ptr_uy(uy_ptr), engine(random_seed), T(T), X(X), t(0), red_locations{0} { }
            
            /*!
             * @brief Evolve the system.
             * 
             * Evolve the current state by moving and replicating particles.
             * During internal computations, \f$\mathcal{B}(n,p)\f$ distributed random variables are replaced by np \f$np>\texttt{det_thr}\f$.
             *
             * @param n_steps number of steps to evolve
             * @param det_thr threshold for deterministic approximation
             */
            void evolve(long n_steps = 1, unsigned long det_thr = 1<<20);
            auto cbegin_r() const { return red_locations.cbegin(); }
            auto cend_r() const { return red_locations.cend(); }
            auto cbegin_y() const { return n_yellow.cbegin(); }
            auto cend_y() const { return n_yellow.cend(); }
            //! number of red particles; equal to cend_r() - cbegin_r()
            auto n_r() const { return red_locations.size(); }
            //! total number of yellow particles
            auto n_y() const { return std::accumulate(n_yellow.cbegin(), n_yellow.cend(), 0); }

            //! number of particles at the site \f$x\f$
            auto y_at(int x) const { if (std::map<int, branching_random_walk::ptcl_n>::const_iterator search = n_yellow.find(x); search != n_yellow.end()) return search->second; else return (branching_random_walk::ptcl_n) 0; }
    };
}

#endif