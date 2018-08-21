/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "Shape.hpp"

namespace Lnn {

Shape::Shape(const std::vector<int64_t>& dimensions) :
		dimensions(dimensions) {
}

std::vector<int64_t> Shape::toVector() const {
	return dimensions;
}
}

