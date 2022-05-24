#ifndef INTEGRATE_TASK_4_OPTIONS_PARSER_H
#define INTEGRATE_TASK_4_OPTIONS_PARSER_H

#include <boost/program_options.hpp>
#include <string>
#include <exception>
#include <stdexcept>
#include "langermann.h"
#include "conf.h"

struct OptionsParseException : public std::runtime_error {
    using runtime_error::runtime_error;
};

struct WrongNumberOfCmdArgs : public OptionsParseException {
    using OptionsParseException::OptionsParseException;
};

struct CantOpenConfigFile : public OptionsParseException {
    using OptionsParseException::OptionsParseException;
};

struct WrongConfigFileOptions : public OptionsParseException {
    using OptionsParseException::OptionsParseException;
};

struct UnrecognizedCmdOption : public OptionsParseException {
    using OptionsParseException::OptionsParseException;
};

class command_line_options_t {
public:
    command_line_options_t();
    command_line_options_t(int ac, char **av);

    //! Explicit is better than implicit:
    command_line_options_t(const command_line_options_t&) = default;
    command_line_options_t& operator=(const command_line_options_t&) = delete;
    command_line_options_t(command_line_options_t&&) = default;
    command_line_options_t& operator=(command_line_options_t&&) = delete;
    ~command_line_options_t() = default;

    [[nodiscard]] Config get_config() const;

    void parse(int ac, char **av);
private:
    Config config;
    void validate_config() const;

    boost::program_options::variables_map var_map{};
    boost::program_options::options_description cmd_options{"Command Line Options"};
    boost::program_options::positional_options_description cmd_positional_options;
    boost::program_options::options_description config_file_options{"Config File Options"};
};

#endif //INTEGRATE_TASK_4_OPTIONS_PARSER_H

