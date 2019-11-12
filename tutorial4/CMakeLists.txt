add_executable(tutorial4_matrix_multiplication_GPU_and_memory_management
		tutorial4.cc

		data/data_type.h
		data/matrix_data.h
		data/matrix_block_data.h
		data/cuda_matrix_block_data.h

		state/cuda_input_block_state.h
		state/output_state.h
		state/partial_computation_state.h
		state/partial_computation_state_manager.h

		cuda_tasks/cuda_copy_in_gpu.h
		cuda_tasks/cuda_copy_out_gpu.h
		cuda_tasks/cuda_product_task.h

		task/addition_task.h
		task/matrix_row_traversal_task.h
		task/matrix_column_traversal_task.h

		utils/cuda_utils.h

		${tclap_headers}
		)

target_compile_definitions(tutorial4_matrix_multiplication_GPU_and_memory_management PUBLIC
		HH_USE_CUDA)

target_link_libraries(tutorial4_matrix_multiplication_GPU_and_memory_management
		${Hedgehog_LIBRARIES}
		${CUDA_LIBRARIES}
		cuda)

target_include_directories(tutorial4_matrix_multiplication_GPU_and_memory_management PUBLIC
		${Hedgehog_INCLUDE_DIR}
		${CUDA_INCLUDE_DIRS})

CUDA_ADD_CUBLAS_TO_TARGET(tutorial4_matrix_multiplication_GPU_and_memory_management)

