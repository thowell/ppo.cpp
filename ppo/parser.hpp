// Copyright 2024 Taylor Howell

#ifndef PPO_PARSER_HPP_
#define PPO_PARSER_HPP_

#include <string>
#include <unordered_map>
#include <vector>

// parse command-line arguments (from Claude Sonnet 3.5)
class ArgumentParser {
 public:
  ArgumentParser(int argc, char* argv[]);

  // get value based on key; if no key, return default value
  std::string Get(const std::string& key,
                  const std::string& default_value = "");

  // check for key
  bool Has(const std::string& key);

  // print information about parser
  void Help();

  // available keys for parsing
  std::vector<std::string> keys;

 private:
  std::unordered_map<std::string, std::string> args_;
};

#endif  // PPO_PARSER_HPP_
