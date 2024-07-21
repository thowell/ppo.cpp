
// Copyright 2024 Taylor Howell

#ifndef PPO_PPO_HPP_
#define PPO_PPO_HPP_

#include <absl/base/attributes.h>
#include <absl/random/random.h>
#include <torch/torch.h>

#include <algorithm>
#include <string>
#include <vector>

#include "agent.hpp"
#include "batch_environment.hpp"
#include "settings.hpp"
#include "utilities.hpp"

template <typename T>
class ProximalPolicyOptimization {
 public:
  // constructor
  ProximalPolicyOptimization(BatchEnvironment<T>& env,
                             BatchEnvironment<T>& eval_env, Agent& agent,
                             torch::optim::Adam& optimizer, Settings& settings)
      : agent_(agent) {
    env_ = &env;
    eval_env_ = &eval_env;
    optimizer_ = &optimizer;
    settings_ = settings;

    // dimensions
    action_size_ = env_->NumAction();
    observation_size_ = env_->NumObservation();

    // -- simulation storage -- //

    // memory
    action_sim_.resize(settings_.num_steps * settings_.num_envs * action_size_);
    observation_sim_.resize(settings_.num_steps * settings_.num_envs *
                            observation_size_);
    reward_sim_.resize(settings_.num_steps * settings_.num_envs);
    done_sim_.resize(settings_.num_steps * settings_.num_envs);

    next_observation_sim_.resize(settings_.num_envs * observation_size_);
    next_done_sim_.resize(settings_.num_envs);

    reward_eval_.resize(settings_.num_eval_envs);
    next_observation_eval_.resize(settings_.num_eval_envs * observation_size_);
    next_done_eval_.resize(settings_.num_eval_envs);

    // reset
    std::fill(action_sim_.begin(), action_sim_.end(), 0.0);
    std::fill(observation_sim_.begin(), observation_sim_.end(), 0.0);
    std::fill(reward_sim_.begin(), reward_sim_.end(), 0.0);
    std::fill(done_sim_.begin(), done_sim_.end(), 0);

    std::fill(next_observation_sim_.begin(), next_observation_sim_.end(), 0.0);
    std::fill(next_done_sim_.begin(), next_done_sim_.end(), 0);

    std::fill(reward_eval_.begin(), reward_eval_.end(), 0.0);
    std::fill(next_observation_eval_.begin(), next_observation_eval_.end(),
              0.0);
    std::fill(next_done_eval_.begin(), next_done_eval_.end(), 0);

    // -- inference storage -- //

    // memory
    action_inf_ =
        torch::zeros({settings_.num_steps, settings_.num_envs, action_size_},
                     settings_.device_type)
            .to(settings_.device);
    logprobs_inf_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                                 settings_.device_type)
                        .to(settings_.device);
    values_inf_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                               settings_.device_type)
                      .to(settings_.device);

    // observation to tensor
    obs_sim_device_ = torch::zeros({settings_.num_envs, observation_size_},
                                   settings_.device_type)
                          .to(settings_.device);
    obs_sim_device_sim_ = torch::zeros({settings_.num_envs, observation_size_},
                                       settings_.device_sim_type)
                              .to(settings_.device_sim);

    obs_eval_device_ =
        torch::zeros({settings_.num_eval_envs, observation_size_},
                     settings_.device_type)
            .to(settings_.device);
    obs_eval_device_sim_ =
        torch::zeros({settings_.num_eval_envs, observation_size_},
                     settings_.device_sim_type)
            .to(settings_.device_sim);

    // action: device to device_sim
    action_sim_device_sim_ = torch::zeros({settings_.num_envs, action_size_},
                                          settings_.device_sim_type)
                                 .to(settings_.device_sim);
    action_eval_device_sim_ =
        torch::zeros({settings_.num_eval_envs, action_size_},
                     settings_.device_sim_type)
            .to(settings_.device_sim);

    // -- device storage -- //
    next_obs_device_ = torch::zeros({settings_.num_envs, observation_size_},
                                    settings_.device_type)
                           .to(settings_.device);
    next_obs_device_sim_ = torch::zeros({settings_.num_envs, observation_size_},
                                        settings_.device_sim_type)
                               .to(settings_.device_sim);

    next_done_device_ =
        torch::zeros({settings_.num_envs}, torch::dtype(torch::kInt32))
            .to(settings_.device);
    next_done_device_sim_ =
        torch::zeros({settings_.num_envs}, torch::dtype(torch::kInt32))
            .to(settings_.device_sim);

    done_device_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                                torch::dtype(torch::kInt32))
                       .to(settings_.device);
    done_device_sim_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                                    torch::dtype(torch::kInt32))
                           .to(settings_.device_sim);

    reward_device_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                                  settings_.device_type)
                         .to(settings_.device);
    reward_device_sim_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                                      settings_.device_sim_type)
                             .to(settings_.device_sim);

    // -- learn storage -- //

    // advantage estimation
    next_value_ = torch::zeros({1, settings_.num_envs}, settings_.device_type)
                      .to(settings_.device);
    advantages_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                               settings_.device_type)
                      .to(settings_.device);
    lastgaelam_ = torch::zeros({1, settings_.num_envs}, settings_.device_type)
                      .to(settings_.device);
    delta_ = torch::zeros({1, settings_.num_envs}, settings_.device_type)
                 .to(settings_.device);
    nextnonterminal_ =
        torch::zeros({settings_.num_envs}, torch::dtype(torch::kInt32))
            .to(settings_.device);
    nextvalues_ = torch::zeros({1, settings_.num_envs}, settings_.device_type)
                      .to(settings_.device);
    returns_ = torch::zeros({settings_.num_steps, settings_.num_envs},
                            settings_.device_type)
                   .to(settings_.device);

    // batch
    obs_batch_device_sim_ =
        torch::zeros(
            {settings_.num_steps, settings_.num_envs, observation_size_},
            settings_.device_sim_type)
            .to(settings_.device_sim);
    obs_batch_device_ = torch::zeros({settings_.num_steps, settings_.num_envs,
                                      observation_size_},
                                     settings_.device_type)
                            .to(settings_.device);

    // flat batch
    b_obs_ = torch::zeros(
                 {settings_.num_steps * settings_.num_envs, observation_size_},
                 settings_.device_type)
                 .to(settings_.device);
    b_logprobs_ = torch::zeros({settings_.num_steps * settings_.num_envs},
                               settings_.device_type)
                      .to(settings_.device);
    b_actions_ =
        torch::zeros({settings_.num_steps * settings_.num_envs, action_size_},
                     settings_.device_type)
            .to(settings_.device);
    b_advantages_ = torch::zeros({settings_.num_steps * settings_.num_envs},
                                 settings_.device_type)
                        .to(settings_.device);
    b_returns_ = torch::zeros({settings_.num_steps * settings_.num_envs},
                              settings_.device_type)
                     .to(settings_.device);
    b_values_ = torch::zeros({settings_.num_steps * settings_.num_envs},
                             settings_.device_type)
                    .to(settings_.device);

    // minibatch
    mb_obs_ = torch::zeros({settings_.minibatch_size, observation_size_},
                           settings_.device_type)
                  .to(settings_.device);
    mb_action_ = torch::zeros({settings_.minibatch_size, action_size_},
                              settings_.device_type)
                     .to(settings_.device);
    mb_logprobs_ =
        torch::zeros({settings_.minibatch_size}, settings_.device_type)
            .to(settings_.device);
    mb_advantages_ =
        torch::zeros({settings_.minibatch_size}, settings_.device_type)
            .to(settings_.device);

    // -- batch indices -- //
    for (int i = 0; i < settings_.batch_size; i++) {
      b_inds_.push_back(i);
    }

    // -- timers -- //
    timer_sim_.clear();
    timer_eval_.clear();
    timer_gae_.clear();
    timer_learn_.clear();
    timer_iter_.clear();

    // -- evaluation information -- //
    reward_eval_avg_.clear();
  }

  // simulate to collect batch of experience from learning environment
  void Simulate() {
    // simulation timer
    auto start_simulation = std::chrono::steady_clock::now();

    // simulate environment
    for (int step = 0; step < settings_.num_steps; step++) {
      // update simulation storage (next_obs -> obs_t, next_done -> done_t)
      std::copy(next_observation_sim_.begin(), next_observation_sim_.end(),
                observation_sim_.begin() +
                    step * settings_.num_envs * observation_size_);
      std::copy(next_done_sim_.begin(), next_done_sim_.end(),
                done_sim_.begin() + step * settings_.num_envs);

      // policy inference w/o collecting gradient information
      {
        // torch no grad
        torch::NoGradGuard no_grad;

        // observation to tensor
        obs_sim_device_sim_ = torch::from_blob(
            const_cast<T*>(next_observation_sim_.data()),
            {settings_.num_envs, observation_size_}, settings_.device_sim_type);

        // observation: type + device
        obs_sim_device_ =
            obs_sim_device_sim_.to(settings_.device_type).to(settings_.device);

        // policy inference
        // TODO(taylor): inplace operation?
        auto [action, logprob, entropy, value] =
            agent_->GetActionAndValue(obs_sim_device_, {}, settings_.device);

        // cache inference variables
        action_inf_[step] = action;
        logprobs_inf_[step] = logprob.flatten();
        values_inf_[step] = value.flatten();
      }

      // action from device to device_sim
      action_sim_device_sim_ = action_inf_[step]
                                   .to(settings_.device_sim)
                                   .to(settings_.device_sim_type);
      T* action_ptr = action_sim_device_sim_.data_ptr<T>();

      // copy to action to memory
      std::copy(action_ptr, action_ptr + settings_.num_envs * action_size_,
                action_sim_.data() + step * settings_.num_envs * action_size_);

      // step environment
      env_->Step(next_observation_sim_.data(),
                 reward_sim_.data() + step * settings_.num_envs,
                 next_done_sim_.data(), action_ptr);
    }

    // simulation timer
    timer_sim_.push_back(GetDuration<double>(start_simulation));
  }

  // generalized advantage estimation
  void Advantage() {
    // timer
    auto start_gae = std::chrono::steady_clock::now();

    // - sim_device -> device - //

    // next_observation to tensor
    next_obs_device_sim_ = torch::from_blob(
        const_cast<T*>(next_observation_sim_.data()),
        {settings_.num_envs, observation_size_}, settings_.device_sim_type);
    next_obs_device_ =
        next_obs_device_sim_.to(settings_.device_type).to(settings_.device);

    // next_done to tensor
    next_done_device_sim_ =
        torch::from_blob(const_cast<int*>(next_done_sim_.data()),
                         {settings_.num_envs}, torch::dtype(torch::kInt32));
    next_done_device_ = next_done_device_sim_.to(settings_.device);

    // done to tensor
    done_device_sim_ = torch::from_blob(
        const_cast<int*>(done_sim_.data()),
        {settings_.num_steps, settings_.num_envs}, torch::dtype(torch::kInt32));
    done_device_ = done_device_sim_.to(settings_.device);

    // reward to tensor
    reward_device_sim_ = torch::from_blob(
        const_cast<T*>(reward_sim_.data()),
        {settings_.num_steps, settings_.num_envs}, settings_.device_sim_type);
    reward_device_ =
        reward_device_sim_.to(settings_.device_type).to(settings_.device);

    // -- reset -- //
    next_value_ = agent_->GetValue(next_obs_device_).reshape({1, -1});
    advantages_.zero_();
    lastgaelam_.zero_();
    delta_.zero_();
    nextnonterminal_.zero_();
    nextvalues_.zero_();
    returns_.zero_();

    // -- advantages -- //
    {
      // torch no grad
      torch::NoGradGuard no_grad;

      // backward through time
      for (int t = settings_.num_steps - 1; t >= 0; t--) {
        if (t == settings_.num_steps - 1) {
          nextnonterminal_ = 1.0 - next_done_device_;
          nextvalues_ = next_value_;
        } else {
          nextnonterminal_ = 1.0 - done_device_[t + 1];
          nextvalues_ = values_inf_[t + 1];
        }
        delta_ = reward_device_[t] +
                 settings_.gamma * nextvalues_ * nextnonterminal_ -
                 values_inf_[t];
        lastgaelam_ = delta_ + settings_.gamma * settings_.gae_lambda *
                                   nextnonterminal_ * lastgaelam_;
        advantages_[t] = lastgaelam_.flatten();
      }
      returns_ = advantages_ + values_inf_;
    }

    // stop timer
    timer_gae_.push_back(GetDuration<double>(start_gae));
  }

  // improve policy and value function
  void Learn() {
    // -- train policy and value function -- //
    auto start_learn = std::chrono::steady_clock::now();

    // - flatten experience + move sim storage to device - //

    // observation to tensor
    obs_batch_device_sim_ = torch::from_blob(
        const_cast<T*>(observation_sim_.data()),
        {settings_.num_steps, settings_.num_envs, observation_size_},
        settings_.device_sim_type);

    // observation: type + device
    obs_batch_device_ =
        obs_batch_device_sim_.to(settings_.device_type).to(settings_.device);

    // flatten
    b_obs_ = obs_batch_device_.reshape({-1, observation_size_});
    b_logprobs_ = logprobs_inf_.reshape({-1});
    b_actions_ = action_inf_.reshape({-1, action_size_});
    b_advantages_ = advantages_.reshape({-1});
    b_returns_ = returns_.reshape({-1});
    b_values_ = values_inf_.reshape({-1});

    // -- learning epochs -- //
    for (int epoch = 0; epoch < settings_.update_epochs; epoch++) {
      // randomly shuffle indices
      // https://stackoverflow.com/a/6926473
      auto rd = std::random_device{};
      auto rng = std::default_random_engine{rd()};
      std::shuffle(b_inds_.begin(), b_inds_.end(), rng);

      // approx kl (initialize)
      at::Tensor approx_kl;

      // minibatches
      for (int start = 0; start < settings_.batch_size;
           start += settings_.minibatch_size) {
        // indices
        std::vector<int> mb_inds(
            b_inds_.begin() + start,
            b_inds_.begin() + start + settings_.minibatch_size);
        torch::Tensor idx = torch::tensor(mb_inds, torch::dtype(torch::kInt32))
                                .to(settings_.device);

        // minibatch elements
        mb_obs_ = b_obs_.index({idx, torch::indexing::Slice()});
        mb_action_ = b_actions_.index({idx, torch::indexing::Slice()});
        mb_logprobs_ = b_logprobs_.index({idx});
        mb_advantages_ = b_advantages_.index({idx});

        // per-minibatch normalization
        if (settings_.norm_adv) {
          mb_advantages_ = (mb_advantages_ - mb_advantages_.mean()) /
                           (mb_advantages_.std() + 1.0e-8);
        }

        // policy inference w/ gradient information
        auto [newaction, newlogprob, newentropy, newvalue] =
            agent_->GetActionAndValue(mb_obs_, mb_action_, settings_.device);

        // ratio
        at::Tensor logratio = newlogprob - mb_logprobs_;
        at::Tensor ratio = logratio.exp();

        // KL divergence
        {
          // torch no grad
          torch::NoGradGuard no_grad;

          // calculate approx_kl http://joschu.net/blog/kl-approx.html
          approx_kl = ((ratio - 1.0f) - logratio).mean();
        }

        // policy loss
        at::Tensor pg_loss1 = -mb_advantages_ * ratio;
        at::Tensor pg_loss2 =
            -mb_advantages_ * torch::clamp(ratio, 1.0f - settings_.clip_coef,
                                           1.0f + settings_.clip_coef);
        at::Tensor pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        // value loss
        newvalue = newvalue.flatten();

        at::Tensor v_loss;
        at::Tensor vrdiff = newvalue - b_returns_.index({idx});
        at::Tensor v_loss_unclipped = vrdiff * vrdiff;
        if (settings_.clip_vloss) {
          at::Tensor v_clipped =
              b_values_.index({idx}) +
              torch::clamp(newvalue - b_values_.index({idx}),
                           -settings_.clip_coef, settings_.clip_coef);

          at::Tensor vcrdiff = v_clipped - b_returns_.index({idx});
          at::Tensor v_loss_clipped = vcrdiff * vcrdiff;

          at::Tensor v_loss_max = torch::max(v_loss_unclipped, v_loss_clipped);
          v_loss = 0.5 * v_loss_max.mean();
        } else {
          v_loss = 0.5 * v_loss_unclipped.mean();
        }

        // entropy loss
        at::Tensor entropy_loss = newentropy.mean();

        // total loss
        at::Tensor loss = pg_loss - settings_.ent_coef * entropy_loss +
                          settings_.vf_coef * v_loss;

        // -- gradient update -- //
        optimizer_->zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(agent_->parameters(),
                                          settings_.max_grad_norm);
        optimizer_->step();
      }

      // kl divergence threshold
      if (approx_kl.item<float>() > settings_.kl_threshold) {
        break;
      }
    }

    // timers
    timer_learn_.push_back(GetDuration<double>(start_learn));
  }

  void Evaluate(int iteration, int global_step) {
    // evaluation timer
    auto start_eval = std::chrono::steady_clock::now();

    // do not collect gradient
    torch::NoGradGuard no_grad;

    // copy observation statistics
    eval_env_->ObservationStatistics(*env_);

    // reset environment
    eval_env_->Reset(next_observation_eval_.data());
    std::fill(next_done_eval_.begin(), next_done_eval_.end(), 0);

    // episode info
    int episodes = 0;
    T episode_reward = 0.0;

    // simulate max_eval_steps
    for (int step = 0; step < settings_.max_eval_steps; step++) {
      // observation to tensor
      obs_eval_device_sim_ =
          torch::from_blob(const_cast<T*>(next_observation_eval_.data()),
                           {settings_.num_eval_envs, observation_size_},
                           settings_.device_sim_type);

      // observation: type + device
      obs_eval_device_ =
          obs_eval_device_sim_.to(settings_.device_type).to(settings_.device);

      // policy inference
      auto [action, logprob, entropy, value] =
          agent_->GetActionAndValue(obs_eval_device_, {}, settings_.device);

      // action from device to device_sim
      action_eval_device_sim_ =
          action.to(settings_.device_sim).to(settings_.device_sim_type);
      T* action_ptr = action_eval_device_sim_.data_ptr<T>();

      // step environment
      eval_env_->Step(next_observation_eval_.data(), reward_eval_.data(),
                      next_done_eval_.data(), action_ptr);

      // update episodes info
      episodes +=
          std::accumulate(next_done_eval_.begin(), next_done_eval_.end(), 0);
      episode_reward +=
          std::accumulate(reward_eval_.begin(), reward_eval_.end(), 0.0);
    }

    // timer
    timer_eval_.push_back(GetDuration<double>(start_eval));

    // info
    reward_eval_avg_.push_back(episode_reward / episodes);
    double total_sim_time = std::accumulate(
        timer_sim_.end() - settings_.num_iter_per_eval, timer_sim_.end(), 0.0);
    double total_gae_time = std::accumulate(
        timer_gae_.end() - settings_.num_iter_per_eval, timer_gae_.end(), 0.0);
    double total_learn_time =
        std::accumulate(timer_learn_.end() - settings_.num_iter_per_eval,
                        timer_learn_.end(), 0.0);
    double total_iter_time =
        std::accumulate(timer_iter_.end() - settings_.num_iter_per_eval,
                        timer_iter_.end(), 0.0);

    // print stats
    std::cout << " iteration (" << iteration << "/" << settings_.num_iterations
              << "):\n  reward (avg): " << reward_eval_avg_.back() << "\n"
              << "  time (avg):  "
              << " | sim: " << total_sim_time / settings_.num_iter_per_eval
              << " | GAE: " << total_gae_time / settings_.num_iter_per_eval
              << " | learn: " << total_learn_time / settings_.num_iter_per_eval
              << " | iter: " << total_iter_time / settings_.num_iter_per_eval
              << "\n  time (total):"
              << " | sim: " << total_sim_time << " | GAE: " << total_gae_time
              << " | learn: " << total_learn_time
              << " | iter: " << total_iter_time
              << " | eval: " << timer_eval_.back()
              << "\n  steps / second (avg): "
              << int(settings_.num_iter_per_eval * settings_.batch_size /
                     total_sim_time)
              << "\n  global steps: " << global_step << std::endl;

    // visualize
    if (settings_.visualize) {
      RunningStatistics<T> obs_stats = env_->GetObservationStatistics();
      eval_env_->Visualize(agent_, obs_stats, settings_.max_eval_steps,
                           settings_.device, settings_.device_sim,
                           settings_.device_type, settings_.device_sim_type);
    }

    // checkpoint
    if (!settings_.checkpoint.empty()) {
      // checkpoint path
      std::string iter_str =
          std::to_string(iteration) + "_" +
          std::to_string(static_cast<int>(std::floor(reward_eval_avg_.back())));
      std::filesystem::path cwd = std::filesystem::current_path();
      std::string checkpoint_dir = cwd.string() + "/../checkpoint";

      // create checkpoint directory
      if (!std::filesystem::is_directory(checkpoint_dir) ||
          !std::filesystem::exists(checkpoint_dir)) {
        std::filesystem::create_directory(checkpoint_dir);
      }

      std::string checkpoint_name = settings_.checkpoint + "_" + iter_str;
      std::string checkpoint_path_agent =
          checkpoint_dir + "/" + checkpoint_name + ".pt";
      std::string checkpoint_path_obs_stats =
          checkpoint_dir + "/" + checkpoint_name + ".stats";

      // save agent
      torch::save(agent_, checkpoint_path_agent);
      std::cout << "  agent saved: " << checkpoint_name + ".pt" << std::endl;

      // save observation statistics
      RunningStatistics<T> obs_stats = env_->GetObservationStatistics();
      obs_stats.Save(checkpoint_path_obs_stats);
      std::cout << "  observation statistics saved: "
                << checkpoint_name + ".stats" << std::endl;
    }
  }

  // run proximal policy optimization algorithm to learn policy and value
  // function
  void Run() {
    // start timer
    auto start_ppo = std::chrono::steady_clock::now();

    // reset environment
    env_->Reset(next_observation_sim_.data());
    std::fill(next_done_sim_.begin(), next_done_sim_.end(), 0);

    // run
    int global_step = 0;
    for (int iteration = 1; iteration < settings_.num_iterations + 1;
         iteration++) {
      // iteration timer
      auto start_iteration = std::chrono::steady_clock::now();

      // learning rate schedule
      if (settings_.anneal_lr) {
        float frac = 1.0f - (iteration - 1.0f) / settings_.num_iterations;
        float lr_now = frac * settings_.learning_rate;
        optimizer_->param_groups()[0].options().set_lr(lr_now);
      }

      // simulate to collect experience
      Simulate();
      global_step += settings_.batch_size;

      // generalized advantage estimation
      Advantage();

      // train policy and value function
      Learn();

      // iteration timer
      timer_iter_.push_back(GetDuration<double>(start_iteration));

      // evaluation
      if (iteration % settings_.num_iter_per_eval == 0) {
        Evaluate(iteration, global_step);
      }
    }

    // info
    std::cout << "\ntotal time: " << GetDuration<double>(start_ppo)
              << std::endl;
    std::cout << "\ntotal steps: " << global_step << std::endl;
  }

 private:
  BatchEnvironment<T>* env_;
  BatchEnvironment<T>* eval_env_;
  Agent agent_;
  torch::optim::Adam* optimizer_;
  Settings settings_;

  // -- dimensions -- //
  int action_size_;
  int observation_size_;

  // -- simulation storage -- //

  // memory
  std::vector<T> action_sim_;
  std::vector<T> observation_sim_;
  std::vector<T> reward_sim_;
  std::vector<int> done_sim_;

  std::vector<T> next_observation_sim_;
  std::vector<int> next_done_sim_;

  std::vector<T> reward_eval_;
  std::vector<T> next_observation_eval_;
  std::vector<int> next_done_eval_;

  // -- inference storage -- //

  // memory
  at::Tensor action_inf_;
  at::Tensor logprobs_inf_;
  at::Tensor values_inf_;

  // observation to tensor
  at::Tensor obs_sim_device_;
  at::Tensor obs_sim_device_sim_;

  at::Tensor obs_eval_device_;
  at::Tensor obs_eval_device_sim_;

  // action: device to device_sim
  at::Tensor action_sim_device_sim_;
  at::Tensor action_eval_device_sim_;

  // -- device storage -- //
  at::Tensor next_obs_device_;
  at::Tensor next_obs_device_sim_;

  at::Tensor next_done_device_;
  at::Tensor next_done_device_sim_;

  at::Tensor done_device_;
  at::Tensor done_device_sim_;

  at::Tensor reward_device_;
  at::Tensor reward_device_sim_;

  // -- learn storage -- //

  // advantage estimation
  at::Tensor next_value_;
  at::Tensor advantages_;
  at::Tensor lastgaelam_;
  at::Tensor delta_;
  at::Tensor nextnonterminal_;
  at::Tensor nextvalues_;
  at::Tensor returns_;

  // batch
  at::Tensor obs_batch_device_sim_;
  at::Tensor obs_batch_device_;

  // flat batch
  at::Tensor b_obs_;
  at::Tensor b_logprobs_;
  at::Tensor b_actions_;
  at::Tensor b_advantages_;
  at::Tensor b_returns_;
  at::Tensor b_values_;

  // minibatch
  at::Tensor mb_obs_;
  at::Tensor mb_action_;
  at::Tensor mb_logprobs_;
  at::Tensor mb_advantages_;

  // -- batch indices -- //
  std::vector<int> b_inds_;

  // -- timers -- //
  std::vector<double> timer_sim_;
  std::vector<double> timer_eval_;
  std::vector<double> timer_gae_;
  std::vector<double> timer_learn_;
  std::vector<double> timer_iter_;

  // -- evaluation information -- //
  std::vector<T> reward_eval_avg_;
};

#endif  // PPO_PPO_HPP_
