add_executable(tutorial2_hadamard_product_with_state
		tutorial2.cc
		data/matrix_block_data.h
		data/matrix_data.h
		data/data_type.h
		data/triplet_matrix_block_data.h
		state/block_state.h
		task/hadamard_product.h
		task/matrix_row_traversal_task.h
		${tclap_headers} )

target_link_libraries(tutorial2_hadamard_product_with_state ${Hedgehog_LIBRARIES})
target_include_directories(tutorial2_hadamard_product_with_state PUBLIC ${Hedgehog_INCLUDE_DIR})