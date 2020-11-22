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
package org.zxg.ai.lnn.tuple;

/**
 * @author <a href="mailto:xianguang.zhou@outlook.com">Xianguang Zhou</a>
 */
public class Tuple {

	public static IntTuple2 it(int e0, int e1) {
		return new IntTuple2(e0, e1);
	}

	public static IntTuple3 it(int e0, int e1, int e2) {
		return new IntTuple3(e0, e1, e2);
	}

	public static FloatTuple2 ft(float e0, float e1) {
		return new FloatTuple2(e0, e1);
	}

	public static IntTuple2 t(int e0, int e1) {
		return new IntTuple2(e0, e1);
	}

	public static IntTuple3 t(int e0, int e1, int e2) {
		return new IntTuple3(e0, e1, e2);
	}

	public static FloatTuple2 t(float e0, float e1) {
		return new FloatTuple2(e0, e1);
	}
}
