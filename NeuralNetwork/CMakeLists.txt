set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(NeuralNetwork "NeuralNetwork")

set(CMAKE_AUTOMOC ON)

set(
    SRC
    NeuralNetwork
    neuralnetwork.h
    neuralnetwork.cpp
)

add_library(${NeuralNetwork} SHARED ${SRC})
target_include_directories(${NeuralNetwork} PUBLIC ../Layer)

find_package(Qt6 REQUIRED COMPONENTS Core)
target_link_libraries(${NeuralNetwork} PRIVATE Qt6::Core Layer)
