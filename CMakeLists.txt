# Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
# 
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

project(light_neural_network)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

include_directories(
	$ENV{TVM_HOME}/include
	$ENV{TVM_HOME}/dlpack/include
	$ENV{TVM_HOME}/dmlc-core/include)

find_library(TVM_RUNTIME_LIBRARY tvm_runtime $ENV{TVM_HOME}/build)
if(${TVM_RUNTIME_LIBRARY} STREQUAL "TVM_RUNTIME_LIBRARY-NOTFOUND")
	message(FATAL_ERROR "TVM runtime library is not found.")
endif()

install(FILES
	Context.hpp
	DataType.hpp
	Shape.hpp
	NDArray.hpp
    DESTINATION include/light_neural_network)

aux_source_directory(. SRC_LIST)

set(LIBRARY_NAME ${PROJECT_NAME})
set(STATIC_LIBRARY_NAME "${LIBRARY_NAME}_static")

add_library(${STATIC_LIBRARY_NAME} STATIC ${SRC_LIST})
target_link_libraries(${STATIC_LIBRARY_NAME}
	${TVM_RUNTIME_LIBRARY})

add_library(${LIBRARY_NAME} SHARED ${SRC_LIST})
target_link_libraries(${LIBRARY_NAME} ${STATIC_LIBRARY_NAME})

install(TARGETS ${STATIC_LIBRARY_NAME} ${LIBRARY_NAME}
    LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

