/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.autograd;

import org.zxg.ai.lnn.LnnException;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class RecordException extends LnnException {

	private static final long serialVersionUID = 1L;

	public RecordException() {
	}

	public RecordException(String message) {
		super(message);
	}

	public RecordException(Throwable cause) {
		super(cause);
	}

	public RecordException(String message, Throwable cause) {
		super(message, cause);
	}

	public RecordException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}
}
