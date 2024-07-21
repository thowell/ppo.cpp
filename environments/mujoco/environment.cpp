// Copyright 2024 Taylor Howell

#include "environment.hpp"

#include <mujoco/mujoco.h>

#include <iostream>
#include <string>
#include <tuple>

// get sensor data using string
double* SensorByName(const mjModel* m, const mjData* d,
                     const std::string& name) {
  int id = mj_name2id(m, mjOBJ_SENSOR, name.c_str());
  if (id == -1) {
    std::cerr << "sensor \"" << name << "\" not found.\n";
    return nullptr;
  } else {
    return d->sensordata + m->sensor_adr[id];
  }
}

// check mjData for warnings, return true if any warnings
bool CheckWarnings(mjData* data) {
  bool warnings_found = false;
  for (int i = 0; i < mjNWARNING; i++) {
    if (data->warning[i].number > 0) {
      // reset
      data->warning[i].number = 0;

      // return failure
      warnings_found = true;
    }
  }
  return warnings_found;
}

// load model from path
mjModel* LoadTestModel(std::string path) {
  // load model
  char loadError[1024] = "";
  mjModel* model = mj_loadXML(path.c_str(), nullptr, loadError, 1000);
  if (loadError[0]) std::cerr << "load error: " << loadError << '\n';

  return model;
}
