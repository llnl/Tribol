blt_add_executable(
  NAME tribol_rewritten_benchmark
  SOURCES driver/RewrittenDriver.cpp
  OUTPUT_DIR ${CMAKE_BINARY_DIR}/benchmarks
  DEPENDS_ON tribol::core
  FOLDER tribol/benchmarks)

target_include_directories(tribol_rewritten_benchmark PRIVATE driver)
target_compile_features(tribol_rewritten_benchmark PRIVATE cxx_std_20)

if(TRIBOL_ENABLE_TESTS)
  blt_add_test(
    NAME tribol_benchmark_smoke
    COMMAND tribol_rewritten_benchmark --case penalty-2d --size 1 --warmups 0
            --iterations 1)

  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)
  add_test(
    NAME tribol_benchmark_orchestration
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s
            ${PROJECT_SOURCE_DIR}/scripts/benchmarks/tests -v)
  set_tests_properties(tribol_benchmark_orchestration PROPERTIES
                       WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
