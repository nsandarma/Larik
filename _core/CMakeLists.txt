CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(Tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_EXTENSIONS OFF)


find_library(ACCELERATE_FRAMEWORK Accelerate)
include_directories(${CMAKE_SOURCE_DIR}/include)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
ADD_EXECUTABLE(tensor src/main.cc)
# Tautkan framework Accelerate
target_link_libraries(tensor PRIVATE ${ACCELERATE_FRAMEWORK})
add_definitions(-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)
# Atur properti target
target_include_directories(tensor PRIVATE ${CMAKE_SOURCE_DIR}/include)


