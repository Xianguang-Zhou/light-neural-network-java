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
package org.zxg.ai.lnn.opencl;

import java.io.Closeable;

import com.jogamp.opencl.CLDevice;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public abstract class Kernel implements Closeable {

	private CLDevice device;
	private Program program;

	void init(CLDevice device) {
		this.device = device;
		Class<?> type = this.getClass();
		String fileName = type.getSimpleName() + ".cl";
		program = new Program(device.getContext(), type.getResourceAsStream(fileName));
	}

	protected Calling call() {
		return program.call().at(device);
	}

	@Override
	public void close() {
		program.close();
	}
}
