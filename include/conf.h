#ifndef INTEGRATE_TASK_4_CONF_H
#define INTEGRATE_TASK_4_CONF_H

#include <iostream>

struct Config {
    Config() = default;
    Config(const Config&) = default;
    Config& operator=(const Config&) = delete;
    ~Config() = default;

    double abs_err = 0;
    double rel_err = 0;
    double x_start = 0;
    double x_end = 0;
    double y_start = 0;
    double y_end = 0;
    int init_steps_x = 0;
    int init_steps_y = 0;
    int max_iter = 0;
};

#ifdef DEBUG_OUTPUT
std::ostream& operator<<(std::ostream& os, const Config& conf);
#endif

#endif //INTEGRATE_TASK_4_CONF_H
