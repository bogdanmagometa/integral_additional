#include "conf.h"

#ifdef DEBUG_OUTPUT
#include <iostream>

std::ostream& operator<<(std::ostream& os, const Config& conf) {
    os << "abs_err: " << conf.abs_err << std::endl;
    os << "rel_err: " << conf.rel_err << std::endl;
    os << "n_threads: " << conf.n_threads << std::endl;
    os << "x_start: " << conf.x_start << std::endl;
    os << "x_end: " << conf.x_end << std::endl;
    os << "y_start: " << conf.y_start << std::endl;
    os << "y_end: " << conf.y_end << std::endl;
    os << "init_steps_x: " << conf.init_steps_x << std::endl;
    os << "init_steps_y: " << conf.init_steps_y << std::endl;
    os << "max_iter: " << conf.max_iter << std::endl;
    return os;
}
#endif
