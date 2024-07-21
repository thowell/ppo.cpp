// Copyright 2024 Taylor Howell

#ifndef PPO_RUNNING_STATISTICS_HPP_
#define PPO_RUNNING_STATISTICS_HPP_

#include <fstream>
#include <string>
#include <vector>

// running statistics
// https://www.johndcook.com/blog/standard_deviation/
// https://github.com/a-mitani/welford/blob/b7f96b9ad5e803d6de665c7df1cdcfb2a53bddc8/welford/welford.py#L132
template <typename T>
class RunningStatistics {
 public:
  explicit RunningStatistics(int dim) {
    // initialize
    dim_ = dim;
    count_ = 0;
    m_old_.resize(dim_);
    m_new_.resize(dim_);
    s_old_.resize(dim_);
    s_new_.resize(dim_);
    var_.resize(dim_);
    std_.resize(dim_);
  }

  // reset count
  void Reset() { count_ = 0; }

  // update
  void Update(const T* x) {
    // increment count
    unsigned long int old_count = count_;
    count_++;

    // initialize
    if (count_ == 1) {
      for (int i = 0; i < dim_; i++) {
        m_old_[i] = x[i];
        m_new_[i] = x[i];
      }
      std::fill(s_old_.begin(), s_old_.end(), 0.0);
      std::fill(s_new_.begin(), s_new_.end(), 0.0);
    } else {
      for (int i = 0; i < dim_; i++) {
        T delta = x[i] - m_old_[i];
        m_new_[i] = m_old_[i] + delta / count_;
        s_new_[i] = s_old_[i] + delta * delta * old_count / count_;
      }
      m_old_ = m_new_;
      s_old_ = s_new_;
    }
  }

  // identity initialization
  void InitializeIdentity() {
    Reset();
    count_ = 1;

    std::fill(m_old_.begin(), m_old_.end(), 0.0);
    std::fill(m_new_.begin(), m_new_.end(), 0.0);

    std::fill(s_old_.begin(), s_old_.end(), 1.0);
    std::fill(s_new_.begin(), s_new_.end(), 1.0);
  }

  // number of data points
  int Count() { return count_; }

  // mean
  const std::vector<T>& Mean() { return m_new_; }

  // variance
  const std::vector<T>& Variance() {
    for (int i = 0; i < dim_; i++) {
      var_[i] = count_ > 1 ? s_new_[i] / (count_ - 1) : 1.0;
    }
    return var_;
  }

  // standard deviation
  const std::vector<T>& StandardDeviation() {
    // compute variance
    Variance();

    // compute standard deviation
    for (int i = 0; i < dim_; i++) {
      std_[i] = (var_[i] > 1.0e-8) ? std::sqrt(var_[i]) : 1.0;
    }
    return std_;
  }

  // merge running statistics
  void Merge(const RunningStatistics& other) {
    unsigned long int count = this->count_ + other.count_;

    for (int i = 0; i < this->dim_; i++) {
      T delta = this->m_new_[i] - other.m_new_[i];
      T delta2 = delta * delta;

      T mean =
          (this->count_ * this->m_new_[i] + other.count_ * other.m_new_[i]) /
          count;
      T var = this->s_new_[i] + other.s_new_[i] +
              delta2 * this->count_ * other.count_ / count;

      this->m_old_[i] = mean;
      this->m_new_[i] = mean;
      this->s_old_[i] = var;
      this->s_new_[i] = var;
    }
    this->count_ = count;
  }

  // save
  // code from Claude 3.5 Sonnet
  void Save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
      file.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
      file.write(reinterpret_cast<const char*>(&count_), sizeof(count_));

      auto writeVector = [&file](const std::vector<T>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
      };

      writeVector(m_old_);
      writeVector(m_new_);
      writeVector(s_old_);
      writeVector(s_new_);
      writeVector(var_);
      writeVector(std_);

      file.close();
    }
  }

  // load
  // code from Claude 3.5 Sonnet
  void Load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
      file.read(reinterpret_cast<char*>(&count_), sizeof(count_));

      auto readVector = [&file](std::vector<T>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
      };

      readVector(m_old_);
      readVector(m_new_);
      readVector(s_old_);
      readVector(s_new_);
      readVector(var_);
      readVector(std_);

      file.close();
    }
  }

 private:
  int dim_;
  unsigned long int count_;
  std::vector<T> m_old_;
  std::vector<T> m_new_;
  std::vector<T> s_old_;
  std::vector<T> s_new_;
  std::vector<T> var_;
  std::vector<T> std_;
};

#endif  // PPO_RUNNING_STATISTICS_HPP_
