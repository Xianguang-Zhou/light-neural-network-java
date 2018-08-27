/*
 * Copyright (c) 2018, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef LNN_SHAPE_HPP_
#define LNN_SHAPE_HPP_

#include <stdint.h>
#include <vector>

namespace Lnn {

class Shape {
public:
	explicit Shape(const std::vector<int64_t>& dimensions);
	std::vector<int64_t> toVector() const;
private:
	std::vector<int64_t> dimensions;
};
}

#endif

