add_executable(tutorial6_matrix_multiplication_Multi_GPU
        tutorial6.cc

		data/data_type.h
		data/matrix_data.h
		data/matrix_block_data.h
		data/cuda_matrix_block_data.h

		state/cuda_input_block_state.h
        state/partial_computation_state.h
		state/partial_computation_state_manager.h

		cuda_tasks/cuda_copy_in_gpu.h
		cuda_tasks/cuda_copy_out_gpu.h
		cuda_tasks/cuda_product_task.h

		task/addition_task.h
		task/matrix_row_traversal_task.h
		task/matrix_column_traversal_task.h

		utils/cuda_utils.h

		graph/cuda_computation_graph.h

		execution_pipeline/multi_gpu_exec_pipeline.h

		${tclap_headers}
		)

target_compile_definitions(tutorial6_matrix_multiplication_Multi_GPU PUBLIC
		HH_USE_CUDA)

target_link_libraries(tutorial6_matrix_multiplication_Multi_GPU
		${Hedgehog_LIBRARIES}
		cuda
		CUDA::cublas
)

target_include_directories(tutorial6_matrix_multiplication_Multi_GPU PUBLIC
		${Hedgehog_INCLUDE_DIR}
)
