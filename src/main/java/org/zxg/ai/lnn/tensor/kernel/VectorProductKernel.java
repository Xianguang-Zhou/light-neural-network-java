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
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.tensor.Tensor;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class VectorProductKernel extends Kernel {

	public void execute(Tensor left, Tensor right, Tensor result) {
		final int cacheLength = right.shape().get(0);
		Calling c = call();
		c.arg(cacheLength);
		c.in(left.flatData()).in(right.flatData());
		c.cache(CacheArg.Type.FLOAT, cacheLength);
		c.out(result.flatData());
		c.pass(new Range1D(cacheLength));
		c.pass(new Range1D(1));
		c.execute();
	}
}
