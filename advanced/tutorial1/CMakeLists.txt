add_executable(advanced_tutorial1
        tutorial1.cc
        data/data_type.h
        utils/cuda_utils.h
        data/matrix_block_data.h
        data/unified_matrix_block_data.h
        cuda_tasks/cuda_prefetch_in_gpu.h
        cuda_tasks/cuda_product_task.h
        state/unified_input_block_state.h
        graph/unified_computation_graph.h
        execution_pipeline/multi_gpu_exec_pipeline.h
        tasks/addition_task.h
        state/output_state.h
        state/partial_computation_state.h
        state/partial_computation_state_manager.h)

target_compile_definitions(advanced_tutorial1 PUBLIC
        HH_USE_CUDA)

target_link_libraries(advanced_tutorial1
        ${Hedgehog_LIBRARIES}
        ${CUDA_LIBRARIES}
        cuda)

target_include_directories(advanced_tutorial1 PUBLIC
        ${Hedgehog_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS})

CUDA_ADD_CUBLAS_TO_TARGET(advanced_tutorial1)


add_executable(advanced_tutorial1_baseline
        tutorial1-baseline.cc)

target_link_libraries(advanced_tutorial1_baseline
        ${CUDA_LIBRARIES}
        cuda)

target_include_directories(advanced_tutorial1_baseline PUBLIC
        ${CUDA_INCLUDE_DIRS})

CUDA_ADD_CUBLAS_TO_TARGET(advanced_tutorial1_baseline)

