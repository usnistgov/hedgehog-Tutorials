cmake_minimum_required(VERSION 3.12)
project(hedgehog_Tutorials)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

# Need to defined variable Hedgehog_INCLUDE_DIR to the hedgehog path
# hedgehog
find_package(Hedgehog REQUIRED)

# OpenBLAS
find_package(OpenBLAS QUIET)

# Cuda
find_package(CUDA QUIET)
find_package(NVTX QUIET)

if (CUDA_FOUND)
	message(STATUS "CUDA found")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
endif (CUDA_FOUND)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${Hedgehog_CXX_FLAGS})

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic")
endif (NOT MSVC)

#Gather tclap headers
file(GLOB tclap_headers "../../../utils/tclap/*.h")

# Add tutorials target
add_subdirectory(tutorial1)
add_subdirectory(tutorial2)
if (OpenBLAS_FOUND)
	add_subdirectory(tutorial3)
endif (OpenBLAS_FOUND)
if (CUDA_FOUND)
	add_subdirectory(tutorial4)
	add_subdirectory(tutorial5)

	add_subdirectory(advanced/tutorial1)
endif (CUDA_FOUND)