#include <cmath>
#include <cstddef>

#include "langermann.h"

double langermann_function(double x1, double x2) {
    constexpr int m = 5;
    constexpr double a1_params[] = {1, 2, 1, 1, 5};
    constexpr double a2_params[] = {4, 5, 1, 2, 4};
    constexpr double c_params[] = {2, 1, 4, 7, 2};

    double sum = 0;
    for (size_t i = 0; i < m; ++i) {
        double diff1 = x1-a1_params[i];
        double diff2 = x2-a2_params[i];
        double sum_of_squares = diff1*diff1 + diff2*diff2;
        double term = c_params[i] * exp(-M_1_PI * sum_of_squares) * cos(M_PI * sum_of_squares);
        sum += term;
    }
    return -sum;
}
