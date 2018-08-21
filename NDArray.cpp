/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "NDArray.hpp"

namespace Lnn {

typedef tvm::runtime::NDArray Array;

NDArray::NDArray(const Shape& shape, const DataType& dataType,
		const Context& context) :
		_shape(shape), _dataType(dataType), _context(context) {
	Array array = Array::Empty(shape.toVector(), dataType.toDlDataType(),
			context.toDlContext());
	this->array = shared_ptr<Array>(new Array(array));
}

Shape NDArray::shape() const {
	return _shape;
}

DataType NDArray::dataType() const {
	return _dataType;
}

Context NDArray::context() const {
	return _context;
}

int NDArray::copyFromCPU(const void* data, size_t size) {
	return TVMArrayCopyFromBytes(&(array->ToDLPack()->dl_tensor),
			const_cast<void*>(data), size);
}

int NDArray::copyToCPU(void* data, size_t size) const {
	return TVMArrayCopyToBytes(&(array->ToDLPack()->dl_tensor), data, size);
}
}

