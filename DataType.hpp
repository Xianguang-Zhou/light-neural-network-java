/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef LNN_DATATYPE_HPP_
#define LNN_DATATYPE_HPP_

#include <dlpack/dlpack.h>

namespace Lnn {

enum class DataTypeCode {
	Int = kDLInt, UInt = kDLUInt, Float = kDLFloat
};

class DataType {
public:
	explicit DataType(DataTypeCode code, uint8_t bits, uint16_t lanes);
	DLDataType toDlDataType() const;
private:
	DataTypeCode code;
	uint8_t bits;
	uint16_t lanes;
};
}

#endif

