#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

#include "options_parser.h"

namespace po = boost::program_options;

constexpr const char *DEFAULT_PATH_TO_CONFIG_FILE = "./data/config_file.cfg";

using std::vector;

command_line_options_t::command_line_options_t() {
    cmd_options.add_options()
        ("help,h",
                "Show help message")
        ("config,c",
             po::value<std::string>()->default_value(DEFAULT_PATH_TO_CONFIG_FILE),
             "Path to configuration file");
    cmd_positional_options.add("config", 1);
    config_file_options.add_options()
            ("abs_err", po::value<double>(&config.abs_err)->required(),
             "desired absolute error")
            ("rel_err", po::value<double>(&config.rel_err)->required(),
             "desired relative error")
            ("n_threads", po::value<int>(&config.n_threads)->required(),
             "number of threads (0 for contiguous)")
            ("x_start", po::value<double>(&config.x_start)->required(),
             "x coordinate of start of the domain of integration")
            ("x_end", po::value<double>(&config.x_end)->required(),
             "x coordinate of end of the domain of integration")
            ("y_start", po::value<double>(&config.y_start)->required(),
             "y coordinate of start of the domain of integration")
            ("y_end", po::value<double>(&config.y_end)->required(),
             "y coordinate of end of the domain of integration")
            ("init_steps_x", po::value<int>(&config.init_steps_x)->required(),
             "number of points in the initial partition along the x axis")
            ("init_steps_y", po::value<int>(&config.init_steps_y)->required(),
             "number of points in the initial partition along the y axis")
            ("max_iter", po::value<int>(&config.max_iter)->required(),
             "maximum number of iterations");
}

command_line_options_t::command_line_options_t(int ac, char **av):
        command_line_options_t() // Delegate constructor
{
    parse(ac, av);
}

Config command_line_options_t::get_config() const {
    validate_config();
    return config;
}

void command_line_options_t::parse(int ac, char **av) {
    // Process command line options
    try {
        po::parsed_options parsed = po::command_line_parser(ac, av)
                .options(cmd_options).positional(cmd_positional_options).run();
        po::store(parsed, var_map);
    } catch (boost::wrapexcept<boost::program_options::too_many_positional_options_error> &ex) {
        throw WrongNumberOfCmdArgs(ex.what());
    } catch (boost::wrapexcept<boost::program_options::unknown_option> &ex) {
        throw UnrecognizedCmdOption(ex.what());
    }
    if (var_map.count("help")) {
        std::cout << cmd_options << "\n" << config_file_options;
        exit(EXIT_SUCCESS);
    }

    // Process configuration file options
    std::string config_fname{var_map["config"].as<std::string>()};
    std::ifstream ifs{config_fname};
    if (ifs.is_open()) {
        try {
            po::store(po::parse_config_file(ifs, config_file_options), var_map);
        } catch (po::error &ex) {
            throw WrongConfigFileOptions(ex.what());
        }
    } else {
        throw CantOpenConfigFile("Can't open config file " + config_fname);
    }
    try {
        po::notify(var_map);
    } catch (po::error &ex) {
        throw WrongConfigFileOptions(ex.what());
    }
}

void command_line_options_t::validate_config() const {
    if (config.abs_err <= 0) {
        throw WrongConfigFileOptions("Positive abs_err expected, " +
                                        std::to_string(config.abs_err) +
                                        " provided.");
    }
    if (config.rel_err <= 0) {
        throw WrongConfigFileOptions("Positive rel_err expected, " +
                                     std::to_string(config.rel_err) +
                                     " provided.");
    }
    if (config.n_threads < 0) {
        throw WrongConfigFileOptions("Non-negative whole n_threads expected, " +
                                     std::to_string(config.n_threads) +
                                     " provided.");
    }
    if (config.init_steps_x <= 0) {
        throw WrongConfigFileOptions("Positive integer expected for init_steps_x, " +
                                     std::to_string(config.init_steps_x) +
                                     " provided.");
    }
    if (config.init_steps_y <= 0) {
        throw WrongConfigFileOptions("Positive integer expected for init_steps_y, " +
                                     std::to_string(config.init_steps_y) +
                                     " provided.");
    }
    if (config.max_iter <= 0) {
        throw WrongConfigFileOptions("Positive integer expected for max_iter, " +
                                     std::to_string(config.max_iter) +
                                     " provided.");
    }
}
