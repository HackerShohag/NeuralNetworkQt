set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(Layer "Layer")
set(CMAKE_AUTOMOC ON)

set(
    SRC
    Layer
    layer.h
    layer.cpp
)

add_library(${Layer} SHARED ${SRC})

find_package(Qt6 REQUIRED COMPONENTS Core)
target_link_libraries(${Layer} PRIVATE Qt6::Core)
