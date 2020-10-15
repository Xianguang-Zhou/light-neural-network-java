/*
 * Copyright (c) 2020, Xianguang Zhou <xianguang.zhou@outlook.com>. All rights reserved.
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
package org.zxg.ai.lnn.optimizer;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import org.zxg.ai.lnn.autograd.Variable;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public abstract class Optimizer {

	protected List<ParamGroup> paramGroups;
	protected Map<String, Object> defaults;
	protected Map<Object, Object> state = new HashMap<>();

	protected void init(Iterable<ParamGroup> paramGroups, Map<String, Object> defaults) {
		if (null == defaults) {
			throw new NullPointerException();
		}
		this.paramGroups = new LinkedList<>();
		this.defaults = defaults;
		for (ParamGroup group : paramGroups) {
			addParamGroup(group);
		}
	}

	public List<ParamGroup> paramGroups() {
		return paramGroups;
	}

	public Map<String, Object> defaults() {
		return defaults;
	}

	public Map<Object, Object> state() {
		return state;
	}

	public void loadState(Map<Object, Object> state) {
		this.state.putAll(state);
	}

	public void addParamGroup(ParamGroup group) {
		for (Map.Entry<String, Object> option : defaults.entrySet()) {
			group.options.putIfAbsent(option.getKey(), option.getValue());
		}
		this.paramGroups.add(group);
	}

	public Variable step() {
		return step(null);
	}

	public abstract Variable step(Supplier<Variable> lossSupplier);

	public void zeroGradient() {
		for (ParamGroup group : paramGroups) {
			for (Variable param : group.params()) {
				param.zeroGradient();
			}
		}
	}
}
