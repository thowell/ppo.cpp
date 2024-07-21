cmake_minimum_required(VERSION 3.16)

project(
  ppo_cpp
  VERSION 0.1.0
  DESCRIPTION "Proximal Policy Optimization"
  HOMEPAGE_URL "https://github.com/thowell/ppo.cpp"
)

# LibTorch installation
# code from: Claude 3.5 Sonnet
# TODO(taylor): probably a better way to install LibTorch
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build/libtorch/share/cmake/Torch/TorchConfig.cmake")
  message(STATUS "LibTorch is already installed. Skipping download.")
else()
  message(STATUS "LibTorch not found. Downloading...")
  
  # version
  set(LIBTORCH_VERSION "2.3.1")

  if(APPLE)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
  else()
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu121.zip")
  endif()

  # Set the download directory
  set(DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

  # Create the download directory if it doesn't exist
  file(MAKE_DIRECTORY ${DOWNLOAD_DIR})

  # Set the file name for the downloaded archive
  set(LIBTORCH_ARCHIVE "${DOWNLOAD_DIR}/libtorch-shared-with-deps-latest.zip")

  # Download the file (equivalent to wget)
  file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ARCHIVE}
      SHOW_PROGRESS
      STATUS DOWNLOAD_STATUS
      LOG DOWNLOAD_LOG)

  # Check download status
  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)

  if(NOT STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "Error downloading LibTorch: ${ERROR_MESSAGE}")
  endif()

  # Extract the archive (equivalent to unzip)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ARCHIVE}
    WORKING_DIRECTORY ${DOWNLOAD_DIR}
  )

  # Remove the zip file after extraction (optional)
  file(REMOVE ${LIBTORCH_ARCHIVE})

  # Fix for Apple ARM Torch install
  if(APPLE)
    execute_process(
      COMMAND install_name_tool
          -add_rpath "/opt/homebrew/opt/libomp/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/build/libtorch/lib/libtorch_cpu.dylib"
      RESULT_VARIABLE INSTALL_NAME_TOOL_RESULT
    )

    # Check if the command was successful
    if(NOT INSTALL_NAME_TOOL_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to run install_name_tool. Error code: ${INSTALL_NAME_TOOL_RESULT}")
    else()
      message(STATUS "Successfully added rpath to libtorch_cpu.dylib")
    endif()
  endif()
endif()

# find PyTorch
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/build/libtorch" CACHE PATH "Path to LibTorch")
find_package(Torch REQUIRED)

# Global compilation settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For LLVM tooling

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(FindOrFetch)

# MuJoCo
set(MUJOCO_GIT_TAG
    3.1.6
    CACHE STRING "Git revision for MuJoCo."
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  mujoco
  LIBRARY_NAME
  mujoco
  GIT_REPO
  https://github.com/google-deepmind/mujoco.git
  GIT_TAG
  ${MUJOCO_GIT_TAG}
  TARGETS
  mujoco::mujoco
  mujoco::platform_ui_adapter
  EXCLUDE_FROM_ALL
)

# Abseil
set(PPO_CPP_DEP_VERSION_abseil
    fb3621f4f897824c0dbe0615fa94543df6192f30 # LTS 20230802.1
    CACHE STRING "Version of `abseil` to be fetched."
    FORCE
)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
# ABSL_ENABLE_INSTALL is needed for
# https://github.com/protocolbuffers/protobuf/issues/12185#issuecomment-1594685860
set(ABSL_ENABLE_INSTALL ON)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  absl
  LIBRARY_NAME
  abseil-cpp
  GIT_REPO
  https://github.com/abseil/abseil-cpp.git
  GIT_TAG
  ${PPO_CPP_DEP_VERSION_abseil}
  TARGETS
  absl::core_headers
  EXCLUDE_FROM_ALL
)

add_subdirectory(ppo)
add_subdirectory(environments/mujoco)

# run PPO examples
add_executable(run run.cpp)

target_link_libraries(
  run
  PUBLIC
  libmjenv
  mujoco::mujoco
  libppo
  torch
)
