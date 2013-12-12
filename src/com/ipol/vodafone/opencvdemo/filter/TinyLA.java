package com.ipol.vodafone.opencvdemo.filter;

import java.util.List;

import org.opencv.core.Point;

public class TinyLA {

	public static float perimeter(List<Point> a) {
		float sum = 0, dx, dy;

		for (int i = 0; i < a.size(); i++) {
			int i2 = (i + 1) % a.size();

			dx = (float) (a.get(i).x - a.get(i2).x);
			dy = (float) (a.get(i).y - a.get(i2).y);

			sum += Math.sqrt(dx * dx + dy * dy);
		}

		return sum;
	}

}
