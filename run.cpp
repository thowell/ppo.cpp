// Copyright 2024 Taylor Howell

#include <filesystem>
#include <iostream>

#include "environments/mujoco/environment.hpp"
#include "environments/mujoco/humanoid.hpp"
#include "ppo/agent.hpp"
#include "ppo/batch_environment.hpp"
#include "ppo/parser.hpp"
#include "ppo/ppo.hpp"
#include "ppo/running_statistics.hpp"
#include "ppo/settings.hpp"

// data type for MuJoCo
using env_type = mjtNum;  // mjtNum

int main(int argc, char* argv[]) {
  std::cout << "Proximal Policy Optimization (ppo.cpp)" << std::endl;

  // parse settings from command line
  ArgumentParser parser(argc, argv);

  // check for environment
  if (!parser.Has("env")) {
    std::cout << "Environment must be specified: --env" << std::endl;
    return 1;
  }

  // get parser info
  if (parser.Has("help")) {
    parser.Help();
    return 1;
  }

  // settings
  Settings settings;        // default
  settings.Parse(&parser);  // update from parser
  settings.Update();        // update dependent settings

  // environments
  std::cout << "Environment: " << parser.Get("env") << std::endl;

  // learning environment
  MuJoCoEnvironment<env_type>* mj_env;
  MuJoCoEnvironment<env_type>* mj_eval_env;

  if (parser.Get("env") == "humanoid") {
    mj_env = new NVHumanoidEnvironment<env_type>();
    mj_eval_env = new NVHumanoidEnvironment<env_type>();
  } else {
    std::cout << "Invalid environment" << std::endl;
    return 1;
  }

  BatchEnvironment env = BatchEnvironment(
      *mj_env, settings.num_envs, settings.num_thread, true,
      settings.normalize_observation, settings.normalize_reward);
  BatchEnvironment eval_env =
      BatchEnvironment(*mj_env, settings.num_eval_envs, settings.num_thread,
                       true, settings.normalize_observation, false);

  // agent
  Agent agent(env.NumObservation(), env.NumAction());

  // load agent
  if (parser.Has("load")) {
    // checkpoint path
    std::filesystem::path cwd = std::filesystem::current_path();
    std::string checkpoint_dir = cwd.string() + "/../checkpoint";
    std::string checkpoint_path_agent =
        checkpoint_dir + "/" + parser.Get("load") + ".pt";
    std::string checkpoint_path_obs_stats =
        checkpoint_dir + "/" + parser.Get("load") + ".stats";

    // load agent
    torch::load(agent, checkpoint_path_agent);
    std::cout << "agent loaded: " << parser.Get("load") + ".pt" << std::endl;

    // load observation statistics
    RunningStatistics<env_type> obs_stats = env.GetObservationStatistics();
    obs_stats.Load(checkpoint_path_obs_stats);
    env.ObservationStatistics(obs_stats);
    std::cout << "observation statistics loaded: "
              << parser.Get("load") + ".stats" << std::endl;
  }

  agent->to(settings.device);

  // train policy with PPO
  if (parser.Has("train")) {
    // setting information
    settings.Print();

    // optimizer
    auto optimizer_options = torch::optim::AdamOptions(settings.learning_rate)
                                 .eps(settings.optimizer_eps);
    auto optimizer = torch::optim::Adam(agent->parameters(), optimizer_options);

    // proximal policy optimization
    ProximalPolicyOptimization<env_type> ppo(env, eval_env, agent, optimizer,
                                             settings);

    // train
    std::cout << "Learn:" << std::endl;
    ppo.Run();
  }

  // visualize
  if (parser.Has("visualize")) {
    RunningStatistics<env_type> obs_stat = env.GetObservationStatistics();
    env.Visualize(agent, obs_stat, settings.max_eval_steps, settings.device,
                  settings.device_sim, settings.device_type,
                  settings.device_sim_type);
  }

  return 0;
}
