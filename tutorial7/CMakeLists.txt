add_executable(tutorial7_compile_time_analysis
        tutorial7.cc
        graph/graph_c_int_c_int.h
        graph/graph_int_int.h
        task/task_c_int_c_int.h
        task/task_int_int.h
        task/task_int_int_with_can_terminate.h
        )

target_link_libraries(tutorial7_compile_time_analysis ${Hedgehog_LIBRARIES})
target_include_directories(tutorial7_compile_time_analysis PUBLIC ${Hedgehog_INCLUDE_DIR})