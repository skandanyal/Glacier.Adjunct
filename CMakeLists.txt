cmake_minimum_required(VERSION 3.20)
project(WallTime_Comparision)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)

add_subdirectory(nanobench-4.6.0)

add_executable(exp1 exp1.cpp)
target_include_directories(exp1 PRIVATE /usr/local/include)
target_link_libraries(exp1 PRIVATE armadillo)
target_compile_options(exp1 PUBLIC
        $<$<CONFIG:Debug,GDebug>:-g -O0 -march=native>
        $<$<CONFIG:Release,GRelease,Benchmark,GBenchmark>:-g -O3 -march=native -ffast-math>
)

add_executable(exp2 exp2.cpp)
target_include_directories(exp2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include)
target_link_libraries(exp2 PRIVATE armadillo nanobench)
target_compile_options(exp2 PUBLIC
        $<$<CONFIG:Debug,GDebug>:-g -O0 -march=native>
        $<$<CONFIG:Release,GRelease,Benchmark,GBenchmark>:-g -O3 -march=native -ffast-math>
)

add_executable(main main.cpp)
target_include_directories(main PRIVATE /usr/local/include)
target_link_libraries(main PRIVATE armadillo)
target_compile_options(main PUBLIC
        $<$<CONFIG:Debug,GDebug>:-g -O0 -march=native>
        $<$<CONFIG:Release,GRelease,Benchmark,GBenchmark>:-g -O3 -march=native -ffast-math>
)

add_executable(mlpack_linearRegression_exp1 mlpack_linearRegression/exp1.cpp)
target_include_directories(mlpack_linearRegression_exp1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include)
target_link_libraries(mlpack_linearRegression_exp1 PRIVATE armadillo nanobench)
target_compile_options(mlpack_linearRegression_exp1 PUBLIC
        $<$<CONFIG:Debug,GDebug>:-g -O0 -march=native>
        $<$<CONFIG:Release,GRelease,Benchmark,GBenchmark>:-g -O3 -march=native -ffast-math>
)

add_executable(mlpack_linearRegression_exp2 mlpack_linearRegression/exp2.cpp)
target_include_directories(mlpack_linearRegression_exp2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include)
target_link_libraries(mlpack_linearRegression_exp2 PRIVATE armadillo nanobench)
target_compile_options(mlpack_linearRegression_exp2 PUBLIC
        $<$<CONFIG:Debug,GDebug>:-g -O0 -march=native>
        $<$<CONFIG:Release,GRelease,Benchmark,GBenchmark>:-g -O3 -march=native -ffast-math>
)

