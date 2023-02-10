add_executable(tutorial3_parallel_reduction
		tutorial3.cc
		graph/parallel_reduction_graph.h
		task/reduction_task.h
		task/vector_decomposition.h
		task/final_reduction_task.h)

target_link_libraries(tutorial3_parallel_reduction ${Hedgehog_LIBRARIES})
target_include_directories(tutorial3_parallel_reduction PUBLIC ${Hedgehog_INCLUDE_DIR})