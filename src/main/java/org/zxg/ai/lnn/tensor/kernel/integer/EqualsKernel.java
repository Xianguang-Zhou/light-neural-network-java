/*
 * Copyright (c) 2021, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
package org.zxg.ai.lnn.tensor.kernel.integer;

import org.zxg.ai.lnn.opencl.Calling;
import org.zxg.ai.lnn.opencl.IntArray;
import org.zxg.ai.lnn.opencl.Kernel;
import org.zxg.ai.lnn.opencl.Range1D;
import org.zxg.ai.lnn.opencl.ShortArray;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class EqualsKernel extends Kernel {

	public boolean execute(IntArray left, IntArray right) {
		ShortArray result = new ShortArray(new short[] { 0 });
		Calling c = call();
		c.in(left).in(right).inOut(result);
		c.pass(new Range1D(left.length));
		c.execute();
		return 0 == result.get(0);
	}
}
