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

import org.zxg.ai.lnn.autograd.Variable;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class StatelessLayer extends Layer {

	@Override
	public void loadState(Map<String, Variable> state, boolean strict) {
	}

	@Override
	public Iterable<Entry<String, Variable>> namedParameters(String prefix, boolean recurse) {
		return Collections.emptySet();
	}

	@Override
	public Variable parameter(String name) {
		return null;
	}

	@Override
	public Iterable<Variable> parameters(boolean recurse) {
		return Collections.emptySet();
	}

	@Override
	public void registerParameter(String name, Variable parameter) {
	}

	@Override
	public Map<String, Variable> state(Map<String, Variable> destination, String prefix) {
		return destination;
	}

	@Override
	public void zeroGradient() {
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append(getClass().getSimpleName());
		builder.append("()");
		return builder.toString();
	}
}
