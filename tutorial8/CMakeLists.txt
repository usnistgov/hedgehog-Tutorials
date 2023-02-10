add_executable(tutorial8_customization
        tutorial8.cc

        )

target_link_libraries(tutorial8_customization ${Hedgehog_LIBRARIES})
target_include_directories(tutorial8_customization PUBLIC ${Hedgehog_INCLUDE_DIR})