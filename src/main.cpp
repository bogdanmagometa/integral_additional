// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <memory>

#include "options_parser.h"
#include "conf.h"
#include "errors.h"
#include "stopwatch.h"
#include "integration.h"

using time_point = std::chrono::steady_clock::time_point;

static constexpr int variant = 4;

int main(int argc, char* argv[]) {
    std::unique_ptr<Config> conf_ptr;
    try {
        command_line_options_t command_line_options{argc, argv};
        conf_ptr = std::make_unique<Config>(command_line_options.get_config());
    } catch (CantOpenConfigFile &ex) {
        std::cerr << ex.what() << std::endl;
        return static_cast<int>(Error::CONFIG_FILE_OPEN_ERR);
    } catch (WrongConfigFileOptions &ex) {
        std::cerr << ex.what() << std::endl;
        return static_cast<int>(Error::CONFIG_FILE_CONTENT_ERR);
    } catch (WrongNumberOfCmdArgs &ex) {
        std::cerr << ex.what() << std::endl;
        return static_cast<int>(Error::NUM_CMD_ARGS_ERR);
    } catch (UnrecognizedCmdOption &ex) {
        std::cerr << ex.what() << std::endl;
        return static_cast<int>(Error::UNRECOGNIZED_CMD_OPTION);
    }
#ifdef DEBUG_OUTPUT
    std::cout << *conf_ptr;
#endif

    double calculated_integral;
    double abs_error;
    double rel_error;

    time_point start_timestamp = get_current_time_fenced();
    if (conf_ptr->n_threads == 0) {
        concurrent_integration(*conf_ptr, langermann_function, calculated_integral, abs_error,
                               rel_error);
    } else {
        mt_integration(*conf_ptr, langermann_function, calculated_integral, abs_error, rel_error);
    }

    long long durationUs = to_us(get_current_time_fenced() - start_timestamp);

    std::cout.precision(12);
    std::cout << variant << std::endl;
    std::cout << calculated_integral << std::endl;
    std::cout << abs_error << std::endl;
    std::cout << rel_error << std::endl;
    std::cout << durationUs << std::endl;

    if( ( abs_error > conf_ptr->abs_err ) || ( rel_error > conf_ptr->rel_err ) ) {
        std::cerr << "Requested errors not reached within allowed iterations limit" << std::endl;
        return static_cast<int>(Error::FAILED_TO_REACH_DESIRED_ACCURACY);
    }

    return 0;
}
