/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.zxg.ai.lnn.tensor.kernel;

import org.zxg.ai.lnn.opencl.CacheArg;
import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.FloatArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.opencl.Range2D;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class ProductKernel extends Kernel {

	public void execute(Tensor left, Tensor right, Tensor result) {
		FloatArray resultData = result.flatData();
		final int cacheWidth = resultData.length;
		final int cacheHeight = right.shape().get(0);
		Calling c = call();
		c.arg(left.ndim() - 1).arg(right.ndim()).arg(cacheHeight);
		c.in(left.dimSizes()).in(right.dimSizes()).in(result.dimSizes());
		c.in(left.flatData()).in(right.flatData());
		c.cache(CacheArg.Type.FLOAT, cacheWidth * cacheHeight);
		c.out(resultData);
		c.pass(new Range2D(cacheWidth, cacheHeight));
		c.pass(new Range1D(cacheWidth));
		c.execute();
	}
}
