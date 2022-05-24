#include "integration.h"

#include <vector>
#include <cmath>
#include <limits>
#include <thread>

static double sum_over(double (*func_to_sum)(double, double), double x_start, double x_end, double delta_x,
                double y_start, double y_end, double delta_y);
static void sum_over_with_output_parameter(double (*func_to_sum)(double, double), double x_start, double x_end, double delta_x,
                                           double y_start, double y_end, double delta_y, double &outSum);
static void join_threads(std::vector<std::thread> &threads);
static void single_thread_job(double (*func_to_integrate)(double, double), int first_column,
                       int num_columns_to_integrate, const Config &conf, double delta_x, double delta_y,
                       double &outIntegrationResult);

void concurrent_integration(const Config &conf, double (*func_to_integrate)(double, double),
                            double &outCalculatedIntegral, double &outAbsError, double &outRelError) {
    double integral_val;
    double delta_x = (conf.x_end - conf.x_start) / conf.init_steps_x;
    double delta_y = (conf.y_end - conf.y_start) / conf.init_steps_y;

    integral_val = sum_over(func_to_integrate, conf.x_start, conf.x_end, delta_x,
                            conf.y_start, conf.y_end, delta_y) * delta_x * delta_y;

    int num_iter = 0; // TODO: or 1?
    bool to_continue = true;
    while (to_continue) {
#ifdef DEBUG_OUTPUT
        std::cout << "Integral on iteration #" << num_iter << " = " << integral_val << std::endl;
#endif
        double prev_integral_val = integral_val;

        double sum = sum_over(func_to_integrate, conf.x_start + delta_x / 2.0, conf.x_end,
                              delta_x, conf.y_start, conf.y_end, delta_y);
        delta_x /= 2.0;
        integral_val = delta_x * delta_y * sum + integral_val / 2.0;

        sum = sum_over(func_to_integrate, conf.x_start, conf.x_end, delta_x,
                       conf.y_start + delta_y/ 2.0, conf.y_end, delta_y);
        delta_y /= 2.0;
        integral_val = delta_x * delta_y * sum + integral_val / 2.0;

        outAbsError = std::abs(integral_val - prev_integral_val);

        if (prev_integral_val == 0.0) { // division by 0.0 is UB
            outRelError = std::numeric_limits<double>::infinity();
        } else {
            outRelError = outAbsError / std::abs(prev_integral_val);
        }
        to_continue = false;
        to_continue |= (outAbsError > conf.abs_err);
        to_continue |= (outRelError > conf.rel_err);
        ++num_iter;
        to_continue &= (conf.max_iter > num_iter);
    }
    outCalculatedIntegral = integral_val;
}

void mt_integration(const Config &conf, double (*func_to_integrate)(double, double),
                    double &outCalculatedIntegral, double &outAbsError, double &outRelError) {
    std::vector<std::thread> threads;
    threads.reserve(conf.n_threads);

    int steps_x = conf.init_steps_x;
    int steps_y = conf.init_steps_y;
    double width = conf.x_end - conf.x_start;
    double height = conf.y_end - conf.y_start;
    double delta_x = width / conf.init_steps_x;
    double delta_y = height / conf.init_steps_y;

    // Calculating the integral for the first time
    double cur_start_x = conf.x_start;
    double cur_end_x = conf.x_start;
    std::vector<double> results(conf.n_threads);
    for (int n_thread = 0; n_thread < conf.n_threads; ++n_thread) {
        int num_columns = conf.init_steps_x / conf.n_threads + (n_thread < conf.init_steps_x % conf.n_threads);
        cur_end_x += num_columns * delta_x;
        threads.emplace_back(sum_over_with_output_parameter, func_to_integrate, cur_start_x, cur_end_x, delta_x, conf.y_start,
                             conf.y_end, delta_y, std::ref(results[n_thread]));
        cur_start_x += num_columns * delta_x;
    }
    join_threads(threads);
    outCalculatedIntegral = 0;
    for (double result: results) {
        outCalculatedIntegral += result;
    }
    outCalculatedIntegral *= delta_x * delta_y;

    // Approximating the value of integral by increasing number of partitions
    outAbsError = std::numeric_limits<double>::infinity();
    outRelError = std::numeric_limits<double>::infinity();
    int n_iter = 0;
    while (n_iter < conf.max_iter && !(outAbsError < conf.abs_err && outRelError < conf.rel_err)) {
#ifdef DEBUG_OUTPUT
        std::cout << "Integral on iteration #" << n_iter << " = " << outCalculatedIntegral << std::endl;
#endif
        threads.clear();
        double prev_integral_val = outCalculatedIntegral;

        int n_column = 0;
        for (int n_thread = 0; n_thread < conf.n_threads; ++n_thread) {
            int num_columns_to_integrate = steps_x / conf.n_threads + (n_thread < (steps_x % conf.n_threads));
            threads.emplace_back(single_thread_job, func_to_integrate, n_column, num_columns_to_integrate, conf,
                                 width / steps_x, height / steps_y, std::ref(results[n_thread]));
            n_column += num_columns_to_integrate;
        }
        join_threads(threads);
        double sum = 0;
        for (double result: results) {
            sum += result;
        }
        outCalculatedIntegral = outCalculatedIntegral / 4 + sum;

        steps_x *= 2;
        steps_y *= 2;
        delta_x = width / delta_x;
        delta_y = height / delta_y;

        ++n_iter;

        outAbsError = std::abs(outCalculatedIntegral - prev_integral_val);

        if (prev_integral_val == 0.0) { // division by 0.0 is UB
            outRelError = std::numeric_limits<double>::infinity();
        } else {
            outRelError = outAbsError / std::abs(prev_integral_val);
        }
    }
}

static double sum_over(double (*func_to_sum)(double, double), double x_start, double x_end, double delta_x,
                       double y_start, double y_end, double delta_y) {
    double sum = 0;
    constexpr double epsilon = 1E-10;
    for (double x_coord = x_start; x_end - x_coord > epsilon; x_coord += delta_x) {
        for (double y_coord = y_start; y_end - y_coord > epsilon; y_coord += delta_y) {
            sum += func_to_sum(x_coord, y_coord);
        }
    }
    return sum;
}
static void sum_over_with_output_parameter(double (*func_to_sum)(double, double), double x_start, double x_end, double delta_x,
                                    double y_start, double y_end, double delta_y, double &outSum) {
    outSum = sum_over(func_to_sum, x_start, x_end, delta_x, y_start, y_end, delta_y);
}

static void join_threads(std::vector<std::thread> &threads) {
    for (std::thread &cur_thread: threads) {
        cur_thread.join();
    }
}

static void single_thread_job(double (*func_to_integrate)(double, double), int first_column,
                       int num_columns_to_integrate, const Config &conf, double delta_x, double delta_y,
                       double &outIntegrationResult) {
    outIntegrationResult = 0;
    double x_begin = conf.x_start + first_column * delta_x;
    outIntegrationResult = sum_over(func_to_integrate, x_begin, x_begin + num_columns_to_integrate * delta_x,
                                    delta_x, conf.y_start + delta_y / 2, conf.y_end, delta_y);
    outIntegrationResult += sum_over(func_to_integrate, x_begin + delta_x / 2, x_begin + num_columns_to_integrate * delta_x,
                                     delta_x, conf.y_start, conf.y_end, delta_y / 2);
    outIntegrationResult *= delta_x * delta_y / 4;
}
