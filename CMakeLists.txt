cmake_minimum_required(VERSION 3.16)
project(hedgehog_Tutorials)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

# Need to define variable Hedgehog_INCLUDE_DIR to the hedgehog path
# hedgehog
find_package(Hedgehog REQUIRED)

# OpenBLAS
find_package(OpenBLAS QUIET)

# Cuda
find_package(NVTX QUIET)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${Hedgehog_CXX_FLAGS})

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic")
endif (NOT MSVC)

#Gather tclap headers
file(GLOB tclap_headers "../../../utils/tclap/*.h")

# Add tutorials target
add_subdirectory(tutorial1)
add_subdirectory(tutorial2)
add_subdirectory(tutorial3)
if (OpenBLAS_FOUND)
	add_subdirectory(tutorial4)
endif (OpenBLAS_FOUND)
if (CUDAToolkit_FOUND)
	add_subdirectory(tutorial5)
	add_subdirectory(tutorial6)
	add_subdirectory(advanced/tutorial1)
endif (CUDAToolkit_FOUND)
add_subdirectory(tutorial7)
add_subdirectory(tutorial8)
