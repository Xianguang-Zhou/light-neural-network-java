/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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

	final void init(CLDevice device) {
		this.device = device;
		Class<?> type = this.getClass();
		String fileName = type.getSimpleName() + ".cl";
		program = new Program(device.getContext(), type.getResourceAsStream(fileName));
	}

	protected final Calling call() {
		return program.call().at(device);
	}

	@Override
	public final void close() {
		program.close();
	}
}
