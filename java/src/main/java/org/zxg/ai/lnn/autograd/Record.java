/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.autograd;

import java.io.Closeable;
import java.util.Deque;
import java.util.LinkedList;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Record implements Closeable {

	public class Pause implements Closeable {

		protected Pause() {
			pauseStack.push(this);
		}

		@Override
		public final void close() {
			if (null == pauseStack || pauseStack.peek() != this) {
				throw new RecordException();
			}
			pauseStack.pop();
		}
	}

	private static final ThreadLocal<Record> currentRecord = new ThreadLocal<Record>();

	public static final Record current(Record r) {
		if (r != null) {
			currentRecord.set(r);
		} else {
			currentRecord.remove();
		}
		return r;
	}

	public static final Record current() {
		return currentRecord.get();
	}

	private Deque<Pause> pauseStack = new LinkedList<>();

	public Pause pause() {
		if (null != pauseStack) {
			return new Pause();
		} else {
			throw new RecordException();
		}
	}

	public final boolean isClosed() {
		return null == pauseStack;
	}

	public final void close() {
		pauseStack = null;
	}

	public final boolean isRecording() {
		return null != pauseStack && pauseStack.isEmpty();
	}
}
