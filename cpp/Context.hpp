/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef LNN_CONTEXT_HPP_
#define LNN_CONTEXT_HPP_

#include <dlpack/dlpack.h>

namespace Lnn {

enum class DeviceType {
	CPU = kDLCPU,
	GPU = kDLGPU,
	CPUPinned = kDLCPUPinned,
	OpenCL = kDLOpenCL,
	Metal = kDLMetal,
	VPI = kDLVPI,
	ROCM = kDLROCM
};

class Context {
public:
	explicit Context(DeviceType deviceType, int deviceIndex);
	DLContext toDlContext() const;
private:
	DeviceType deviceType;
	int deviceIndex;
};
}

#endif

