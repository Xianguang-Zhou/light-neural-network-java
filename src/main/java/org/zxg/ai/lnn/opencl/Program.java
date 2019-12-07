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
import java.io.IOException;
import java.io.InputStream;

import org.zxg.ai.lnn.LnnIOException;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLException;
import com.jogamp.opencl.CLProgram;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Program implements Closeable {

	private static CLContext DEFAULT_CONTEXT;
	private static CLDevice DEFAULT_DEVICE;

	public static CLContext defaultContext() {
		if (null == DEFAULT_CONTEXT) {
			synchronized (Program.class) {
				if (null == DEFAULT_CONTEXT) {
					DEFAULT_CONTEXT = CLContext.create();
				}
			}
		}
		return DEFAULT_CONTEXT;
	}

	public static CLDevice defaultDevice() {
		if (null == DEFAULT_DEVICE) {
			DEFAULT_DEVICE = defaultContext().getMaxFlopsDevice();
		}
		return DEFAULT_DEVICE;
	}

	private CLProgram program;

	protected Program() {
	}

	public Program(CLContext ctx) throws LnnIOException, LnnCLException {
		init(ctx);
	}

	public Program(CLContext ctx, String name) throws LnnIOException, LnnCLException {
		init(ctx, name);
	}

	public Program(CLContext ctx, InputStream in) throws LnnIOException, LnnCLException {
		init(ctx, in);
	}

	protected void init(CLContext ctx) throws LnnIOException, LnnCLException {
		init(ctx, this.getClass().getSimpleName() + ".cl");
	}

	protected void init(CLContext ctx, String name) throws LnnIOException, LnnCLException {
		init(ctx, this.getClass().getResourceAsStream(name));
	}

	protected void init(CLContext ctx, InputStream in) throws LnnIOException, LnnCLException {
		try {
			this.program = ctx.createProgram(in);
			this.program.build();
		} catch (IOException e) {
			throw new LnnIOException(e);
		} catch (CLException e) {
			throw new LnnCLException(e);
		}
	}

	public Calling call() {
		return call("run");
	}

	public Calling call(String function) {
		return new Calling(program, function);
	}

	@Override
	public void close() {
		this.program.release();
	}
}
