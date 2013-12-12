package com.ipol.vodafone.opencvdemo;

import android.content.Context;
import android.util.DisplayMetrics;

public enum DisplayManager {
	// Singleton.
	INSTANCE;

	private Context context;

	// Display data.
	private float density;
	private int densityDpi;
	private int displayWidth, displayHeight;
	private int displayRatio;

	public void init(Context context) {
		initDisplayData(context);

		this.context = context;
	}

	private void initDisplayData(Context context) {
		final DisplayMetrics metrics = context.getResources().getDisplayMetrics();
		density = metrics.density;
		densityDpi = metrics.densityDpi;
		displayWidth = metrics.widthPixels;
		displayHeight = metrics.heightPixels;
		displayRatio = displayHeight / displayWidth;
	}

	public float getDensity() {
		return density;
	}

	public int getDensityDpi() {
		return densityDpi;
	}

	public int getDisplayWidth() {
		return displayWidth;
	}

	public int getDisplayHeight() {
		return displayHeight;
	}

	public int getDisplayRatio() {
		return displayRatio;
	}

}
