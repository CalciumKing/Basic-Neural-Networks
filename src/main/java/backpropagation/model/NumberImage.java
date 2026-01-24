package backpropagation.model;

/**
 * A record class representing an image of a number from the dataset.
 *
 * @param pixels 2D float array representing all pixel values in greyscale
 * @param value  actual value of the image, assigned dynamically, not guessed
 */
public record NumberImage(float[][] pixels,
                          int value) {
	/**
	 * Converts image from 2D pixel array into longer 1D array of same values, essentially "flattening" values
	 *
	 * @return 1D array preserving all values of 2D array, 1D array length is width * height of image pixels
	 */
	public float[] to1D() {
		float[] flattened = new float[pixels.length * pixels[0].length];
		int idx = 0;
		for (float[] pixel : pixels) {
			for (int c = 0; c < pixels[0].length; c++) {
				flattened[idx] = pixel[c];
				idx++;
			}
		}
		
		return flattened;
	}
	
	/**
	 * Creates empty array except for actual value.
	 * <p>
	 * Example: {@code 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]}
	 *
	 * @return float array with length 10, representing target value
	 */
	public float[] toTarget() {
		float[] target = new float[10];  // 10 representing digits 0-9
		target[value] = 1;
		return target;
	}
	
	/**
	 * Scales image down by a specified factor,
	 * creating an image with same contents but smaller size.
	 *
	 * @param factor times to scale image down
	 * @return new scaled {@link NumberImage} object
	 */
	public NumberImage scaleDownImage(int factor) {
		int height = pixels.length / factor;
		int width = pixels[0].length / factor;
		float[][] newPixels = new float[height][width];
		
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float sum = 0;
				for (int yi = 0; yi < factor; yi++)
					for (int xi = 0; xi < factor; xi++)
						sum += pixels[(y * factor) + yi][(x * factor) + xi];
				newPixels[y][x] = sum / (factor * factor);
			}
		}
		
		return new NumberImage(newPixels, value);
	}
	
	/**
	 * Prints the image in ASCII format using pixels variable.
	 * <p>
	 * Greyscale values are printed as {@code @$#!;:~-,.} in .1 intervals.
	 */
	public void printASCII() {
		for (float[] floats : pixels) {
			for (float v : floats) {
				char[] symbols = { '.', ',', '-', '~', ':', ';', '!', '#', '$', '@' };
				int index = Math.min((int) (v * 10), 9);  // converts 0.0-1.0 decimal to
				char symbol = symbols[index];
				System.out.print(symbol);
			}
			System.out.println();
		}
	}
}
