/*
 * Copyright (c) 2019, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.zxg.ai.lnn.nn;

import java.util.Collections;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Consumer;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public abstract class Layer implements Component {

	@Override
	public void addComponent(String name, Component component) {
	}

	@Override
	public void apply(Consumer<Component> function) {
		function.accept(this);
	}

	@Override
	public Component child(String name) {
		return null;
	}

	@Override
	public Iterable<Component> children() {
		return Collections.emptySet();
	}

	@Override
	public Iterable<Component> components() {
		return Collections.singleton(this);
	}

	@Override
	public Iterable<Entry<String, Component>> namedChildren() {
		return Collections.emptySet();
	}

	@Override
	public Iterable<Entry<String, Component>> namedComponents(Map<String, Component> memo, String prefix) {
		if (!memo.containsValue(this)) {
			memo.put(prefix, this);
		}
		return memo.entrySet();
	}
}
