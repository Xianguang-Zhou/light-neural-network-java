/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "DataType.hpp"

namespace Lnn {

DataType::DataType(DataTypeCode code, uint8_t bits, uint16_t lanes) :
		code(code), bits(bits), lanes(lanes) {
}

DLDataType DataType::toDlDataType() const {
	DLDataType dlDataType = { .code = static_cast<uint8_t>(this->code), .bits = this->bits, .lanes =
			this->lanes };
	return dlDataType;
}
}

