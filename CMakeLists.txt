cmake_minimum_required(VERSION 3.26)
project(flash_kmeans_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target architectures: GTX 1070 (sm_61), Turing, Ampere, RTX 4090 (sm_89) and
# RTX 5090 (sm_120) plus PTX for future architectures.  Override with the
# CUDAARCHS environment variable (e.g. CUDAARCHS=89 pixi run build).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES
        "61-real;75-real;86-real;89-real;120-real;120-virtual")
  endif()
endif()
message(STATUS "flash-kmeans CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _flash_kmeans_cpp
  STABLE_ABI
  NB_STATIC
  src/bindings.cpp
  src/kmeans_impl.cu
  src/large_n.cu)

target_include_directories(_flash_kmeans_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Place the extension inside the Python package.
install(TARGETS _flash_kmeans_cpp LIBRARY DESTINATION flash_kmeans)
