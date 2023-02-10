add_executable(tutorial4_matrix_multiplication_cycle_resolution
		tutorial4.cc

		data/data_type.h
		data/matrix_block_data.h
		data/matrix_data.h

		state/input_block_state.h
        state/partial_computation_state.h
		state/partial_computation_state_manager.h

		task/addition_task.h
		task/product_task.h
		task/matrix_row_traversal_task.h
		task/matrix_column_traversal_task.h

		${tclap_headers}
		)

target_link_libraries(tutorial4_matrix_multiplication_cycle_resolution ${OpenBLAS_LIBRARIES})
target_include_directories(tutorial4_matrix_multiplication_cycle_resolution PUBLIC ${OpenBLAS_INCLUDE_DIRS})
target_compile_definitions(tutorial4_matrix_multiplication_cycle_resolution PUBLIC -DUSE_OPENBLAS)

target_link_libraries(tutorial4_matrix_multiplication_cycle_resolution ${Hedgehog_LIBRARIES})
target_include_directories(tutorial4_matrix_multiplication_cycle_resolution PUBLIC ${Hedgehog_INCLUDE_DIR})