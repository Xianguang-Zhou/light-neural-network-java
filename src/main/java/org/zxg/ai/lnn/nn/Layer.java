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