add_executable(tutorial1_hadamard_product
		tutorial1.cc
		data/matrix_block_data.h
		data/matrix_data.h
		data/data_type.h
		data/triplet_matrix_block_data.h
		task/hadamard_product.h
		${tclap_headers})

target_link_libraries(tutorial1_hadamard_product ${Hedgehog_LIBRARIES})
target_include_directories(tutorial1_hadamard_product PUBLIC ${Hedgehog_INCLUDE_DIR})