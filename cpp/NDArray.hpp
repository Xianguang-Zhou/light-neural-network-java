/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef LNN_NDARRAY_HPP_
#define LNN_NDARRAY_HPP_

#include <memory>
#include <tvm/runtime/ndarray.h>
#include "Shape.hpp"
#include "DataType.hpp"
#include "Context.hpp"

namespace Lnn {

using std::shared_ptr;

class NDArray {
public:
	explicit NDArray(const Shape& shape, const DataType& dataType,
			const Context& context);
	Shape shape() const;
	DataType dataType() const;
	Context context() const;
	int copyFromCPU(const void* data, size_t size);
	int copyToCPU(void* data, size_t size) const;
	virtual NDArray operator+(const NDArray& other) const;
	virtual NDArray operator-(const NDArray& other) const;
	virtual NDArray operator*(const NDArray& other) const;
	virtual NDArray operator/(const NDArray& other) const;
	virtual NDArray power(const NDArray& other) const;
	virtual NDArray operator%(const NDArray& other) const;
private:
	shared_ptr<tvm::runtime::NDArray> array;
	Shape _shape;
	DataType _dataType;
	Context _context;
};
}

#endif

