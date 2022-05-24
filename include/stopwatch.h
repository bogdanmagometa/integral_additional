#ifndef INTEGRATE_TASK_4_STOPWATCH_H
#define INTEGRATE_TASK_4_STOPWATCH_H

#include <chrono>
#include <atomic>
#include <cassert>

inline std::chrono::steady_clock::time_point get_current_time_fenced() {
    assert(std::chrono::steady_clock::is_steady &&
                "Timer should be steady (monotonic).");
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::steady_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

template<class D>
inline long long to_us(const D& d)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

#endif //INTEGRATE_TASK_4_STOPWATCH_H
