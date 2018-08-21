/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "Context.hpp"

namespace Lnn {

Context::Context(DeviceType deviceType, int deviceIndex) :
		deviceType(deviceType), deviceIndex(deviceIndex) {
}

DLContext Context::toDlContext() const {
	DLContext dlContext = { .device_type =
			static_cast<DLDeviceType>(this->deviceType), .device_id =
			this->deviceIndex };
	return dlContext;
}
}

