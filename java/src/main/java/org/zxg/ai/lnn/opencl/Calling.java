/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.opencl;

import java.nio.Buffer;
import java.util.LinkedList;
import java.util.List;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLException;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Calling {

	private CLProgram program;
	private String function;
	private List<Arg> args = new LinkedList<>();
	private List<Range> ranges = new LinkedList<>();
	private CLDevice device;

	public Calling(CLProgram program, String function) {
		this.program = program;
		this.function = function;
	}

	public Calling arg(Arg arg) {
		this.args.add(arg);
		return this;
	}

	public Calling arg(float number) {
		return arg(new FloatArg(number));
	}

	public Calling arg(double number) {
		return arg(new DoubleArg(number));
	}

	public Calling arg(short number) {
		return arg(new ShortArg(number));
	}

	public Calling arg(int number) {
		return arg(new IntArg(number));
	}

	public Calling arg(long number) {
		return arg(new LongArg(number));
	}

	public Calling arg(Buffer directBuffer, BufferArg.Type type) {
		return arg(new BufferArg(directBuffer, type));
	}

	public Calling in(Buffer directBuffer) {
		return arg(directBuffer, BufferArg.Type.IN);
	}

	public Calling out(Buffer directBuffer) {
		return arg(directBuffer, BufferArg.Type.OUT);
	}

	public Calling inOut(Buffer directBuffer) {
		return arg(directBuffer, BufferArg.Type.IN_OUT);
	}

	public Calling pass(Range range) {
		this.ranges.add(range);
		return this;
	}

	public Calling at(CLDevice device) throws LnnCLException {
		if (!program.getContext().equals(device.getContext())) {
			throw new LnnCLException();
		}
		this.device = device;
		return this;
	}

	public void execute() throws LnnCLException {
		if (ranges.isEmpty() || null == device) {
			throw new NullPointerException();
		}
		try {
			CLCommandQueue queue = device.createCommandQueue();
			try {
				CLKernel kernel;
				synchronized (program) {
					kernel = program.createCLKernel(function);
				}
				try {
					for (Arg arg : args) {
						arg.input(kernel, queue);
					}
					if (ranges.size() > 1) {
						int passIdArgIndex = args.size(), passId = 0;
						for (Range range : ranges) {
							if (0 == passId) {
								kernel.putArg(passId);
							} else {
								kernel.setArg(passIdArgIndex, passId);
							}
							range.putToQueue(queue, kernel);
							passId++;
						}
					} else {
						ranges.get(0).putToQueue(queue, kernel);
					}
					for (Arg arg : args) {
						arg.output(queue);
					}
					queue.finish();
					for (Arg arg : args) {
						arg.close();
					}
				} finally {
					kernel.release();
				}
			} finally {
				queue.release();
			}
		} catch (CLException e) {
			throw new LnnCLException(e);
		}
	}
}
