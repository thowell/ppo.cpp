
// Copyright 2024 Taylor Howell

#ifndef ENVIRONMENTS_MUJOCO_ENVIRONMENT_HPP_
#define ENVIRONMENTS_MUJOCO_ENVIRONMENT_HPP_

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <torch/torch.h>

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "../../ppo/environment.hpp"
#include "../../ppo/running_statistics.hpp"
#include "../../ppo/utilities.hpp"

// MuJoCo environment
template <typename T>
class MuJoCoEnvironment : public Environment<T> {
 public:
  // default constructor
  explicit MuJoCoEnvironment(mjModel* model)
      : Environment<T>(model->nu, model->nq + model->nv),
        model_(mj_copyModel(nullptr, model)),
        data_(mj_makeData(model)),
        ndecimation_(1) {}

  // copy constructor
  MuJoCoEnvironment(const MuJoCoEnvironment<T>& env)
      : Environment<T>(env.NumAction(), env.NumObservation()),
        model_(mj_copyModel(nullptr, env.model_)),
        data_(mj_makeData(env.model_)),
        ndecimation_(env.ndecimation_) {}

  // default destructor
  ~MuJoCoEnvironment() {
    if (data_) {
      mj_deleteData(data_);
    }
    if (model_) {
      mj_deleteModel(model_);
    }
  }

  // visualize agent
  void Visualize(Agent& agent,
                 RunningStatistics<T>& obs_stats, int steps,
                 torch::Device device, torch::Device device_sim,
                 torch::Dtype device_type, torch::Dtype device_sim_type) {
    static mjvCamera cam;   // abstract camera
    static mjvOption opt;   // visualization options
    static mjvScene scn;    // abstract scene
    static mjrContext con;  // custom GPU context

    // MuJoCo data structures
    mjModel* m = this->Model();
    mjData* d = this->Data();
    std::vector<T> action(this->NumAction());  // actions from policy
    std::fill(action.begin(), action.end(), 0.0);
    std::vector<T> observation(
        this->NumObservation());  // observation from environment
    std::vector<T> reward(1);
    std::vector<int> done(1);

    // observation normalization
    std::vector<T> mean = obs_stats.Mean();
    std::vector<T> stddev = obs_stats.StandardDeviation();

    // init GLFW, create window, make OpenGL context current, request v-sync
    if (!glfwInit()) {
      mju_error("Could not initialize GLFW");
    }

    GLFWwindow* window =
        glfwCreateWindow(1200, 900, "MuJoCo Learning Environment", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);

    // ... install GLFW keyboard and mouse callbacks

    // environment reset
    this->Reset(observation.data());

    // normalize observation
    for (int j = 0; j < this->NumObservation(); j++) {
      if (std::abs(stddev[j]) < 1.0e-7) {
        observation[j] = 0.0;
      } else {
        observation[j] = (observation[j] - mean[j]) / (stddev[j] + 1.0e-8);
      }
    }

    // simulate physics
    auto render_start = std::chrono::steady_clock::now();
    T physics_start = d->time;

    // run main loop, target real-time simulation and 60 fps rendering
    T duration = m->opt.timestep * this->NumDecimation() * steps;
    while (!glfwWindowShouldClose(window) && d->time < duration) {
      // simulate physics
      auto start = std::chrono::steady_clock::now();

      // policy inference
      at::Tensor obs = torch::from_blob(
          observation.data(), {1, this->NumObservation()}, device_sim_type);

      auto [action, logprob, entropy, value] =
          agent->GetActionAndValue(obs.to(device_type).to(device), {}, device);

      // step environment
      this->Step(observation.data(), reward.data(), done.data(),
               action.ravel().to(device_sim).to(device_sim_type).data_ptr<T>());

      // normalize observation
      for (int j = 0; j < this->NumObservation(); j++) {
        if (std::abs(stddev[j]) < 1.0e-7) {
          observation[j] = 0.0;
        } else {
          observation[j] = (observation[j] - mean[j]) / (stddev[j] + 1.0e-8);
        }
      }

      if (done[0] == 1) return;

      // track
      if (GetDuration<double>(render_start) > 1.0 / 60.0 &&
          d->time - physics_start >= 1.0 / 60.0) {
        this->LookAt(&cam, d);

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

        // reset render timer
        physics_start = d->time;
        render_start = std::chrono::steady_clock::now();
      }

      // wait
      double timer = GetDuration<double>(start);
      while (timer < m->opt.timestep * this->NumDecimation()) {
        timer = GetDuration<double>(start);
      }
    }

// close GLFW, free visualization storage
// terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
  }

  // set look at position for visualization
  virtual void LookAt(mjvCamera* cam, const mjData* data) const {}

  int NumDecimation() const { return ndecimation_; }
  mjModel* Model() const { return model_; }
  mjData* Data() const { return data_; }
  std::tuple<std::vector<T>, std::vector<T>> ActionLimits() const {
    int nu = model_->nu;
    std::vector<T> lower;
    std::vector<T> upper;
    for (int i = 0; i < nu; i++) {
      lower.push_back(model_->actuator_ctrlrange[2 * i]);
      upper.push_back(model_->actuator_ctrlrange[2 * i + 1]);
    }
    return std::make_tuple(lower, upper);
  }

 protected:
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  int ndecimation_;
};

// -- utilities from github.com/google-deepmind/mujoco_mpc -- //

// get sensor data using string
double* SensorByName(const mjModel* m, const mjData* d,
                     const std::string& name);

// check mjData for warnings, return true if any warnings
bool CheckWarnings(mjData* data);

// load model from path
mjModel* LoadTestModel(std::string path);

#endif  // ENVIRONMENTS_MUJOCO_ENVIRONMENT_HPP_
