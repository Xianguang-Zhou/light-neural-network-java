/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn;

import java.io.IOException;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class LnnIOException extends LnnException {

	private static final long serialVersionUID = 1L;

	public LnnIOException(IOException cause) {
		super(cause);
	}

	public LnnIOException(String message, IOException cause) {
		super(message, cause);
	}

	public LnnIOException(String message, IOException cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}
}
