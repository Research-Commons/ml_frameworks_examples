#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>

using i64 = int64_t;

namespace h5 {

//===--------------------------------------------------------------------===//
//
//                  HighFive - Minimal Command Line Argument Parser
//
// HighFive (h5) is a minimal Cmd line parser with dynamic fields you can set
// for dtypes: i64, double and string with `int` automatic coercion
//
// $Example:
// ---------------------------------------------------
// #define MIN_FRUITS 4
//
// auto ok = register_default("fruits", MIN_FRUITS);
// ---------------------------------------------------
//
// `ok` here is taken as i64 as compile time and can be called from the
// command line like so:
//
// ./main fruits=5
//
// Here is an example usage of h5:
//
// ---------------------------------------------------------------
//  h5::Config cfg;
//  cfg.register_default("epochs", i64(50));
//  cfg.register_default("lr", 0.001);
//  cfg.register_default("threads", i64(4));
//  cfg.register_default("mode", std::string("train"));
//
//  cfg.parse_cli(argc, argv);
//
//  /// Can get as int, i64, double, string
//  int epochs_i = cfg.get<int>("epochs");
//  i64 epochs_l = cfg.get<i64>("epochs");
//  double lr = cfg.get<double>("lr");
//  std::string mode = cfg.get<std::string>("mode");
// ---------------------------------------------------------------
//
//===--------------------------------------------------------------------===//

struct Config {
  using Value = std::variant<i64, double, std::string>;
  std::unordered_map<std::string, Value> values;

  /// Registers a key and its default value
  void register_default(const std::string &key, Value default_val) {
    values[key] = default_val;
  }

  /// Parses CLI args of the form key=value
  void parse_cli(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto eq_pos = arg.find('=');
      if (eq_pos == std::string::npos) {
        std::cerr << "Ignoring malformed argument: " << arg << "\n";
        continue;
      }
      std::string key = arg.substr(0, eq_pos);
      std::string val_str = arg.substr(eq_pos + 1);

      auto it = values.find(key);
      if (it == values.end()) {
        std::cerr << "Unknown key: " << key << "\n";
        continue;
      }

      try {
        if (std::holds_alternative<i64>(it->second)) {
          it->second = static_cast<i64>(std::stoll(val_str));
        } else if (std::holds_alternative<double>(it->second)) {
          it->second = std::stod(val_str);
        } else if (std::holds_alternative<std::string>(it->second)) {
          it->second = val_str;
        }
      } catch (...) {
        std::cerr << "Invalid value for key: " << key << "\n";
      }
    }
  }

  /// Get and automatically convert to T if compatible
  template <typename T> T get(const std::string &key) const {
    auto it = values.find(key);
    if (it == values.end()) {
      throw std::runtime_error("Key not found: " + key);
    }

    const Value &v = it->second;

    if constexpr (std::is_same_v<T, int>) {
      if (std::holds_alternative<i64>(v)) {
        return static_cast<int>(std::get<i64>(v));
      }
      if (std::holds_alternative<double>(v)) {
        return static_cast<int>(std::get<double>(v));
      }
      throw std::runtime_error("Type mismatch: cannot convert to int");
    }

    if constexpr (std::is_same_v<T, i64>) {
      if (std::holds_alternative<i64>(v)) {
        return std::get<i64>(v);
      }
      if (std::holds_alternative<double>(v)) {
        return static_cast<i64>(std::get<double>(v));
      }
      throw std::runtime_error("Type mismatch: cannot convert to i64");
    }

    if constexpr (std::is_same_v<T, double>) {
      if (std::holds_alternative<i64>(v)) {
        return static_cast<double>(std::get<i64>(v));
      }
      if (std::holds_alternative<double>(v)) {
        return std::get<double>(v);
      }
      throw std::runtime_error("Type mismatch: cannot convert to double");
    }

    if constexpr (std::is_same_v<T, std::string>) {
      if (std::holds_alternative<std::string>(v)) {
        return std::get<std::string>(v);
      }
      throw std::runtime_error("Type mismatch: cannot convert to string");
    }

    throw std::runtime_error("Unsupported type requested");
  }
};

} // namespace h5
