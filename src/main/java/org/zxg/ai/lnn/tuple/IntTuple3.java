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
package org.zxg.ai.lnn.tuple;

import java.util.Objects;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class IntTuple3 {

	public final int e0;
	public final int e1;
	public final int e2;

	public IntTuple3(int e0, int e1, int e2) {
		this.e0 = e0;
		this.e1 = e1;
		this.e2 = e2;
	}

	public boolean anyoneLessThan(int value) {
		return e0 < value || e1 < value || e2 < value;
	}

	@Override
	public int hashCode() {
		return Objects.hash(e0, e1, e2);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (!(obj instanceof IntTuple3)) {
			return false;
		}
		IntTuple3 other = (IntTuple3) obj;
		return e0 == other.e0 && e1 == other.e1 && e2 == other.e2;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("(");
		builder.append(e0);
		builder.append(", ");
		builder.append(e1);
		builder.append(", ");
		builder.append(e2);
		builder.append(")");
		return builder.toString();
	}
}
