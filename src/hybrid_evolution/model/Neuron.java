package hybrid_evolution.model;

import java.util.Random;

/**
 * A class that represents a single neuron.
 *
 * @see Layer
 * @see NeuralNetwork
 */
public class Neuron {
	private final float[] weights;  // array of all weights for next layer
	private float value;  // weights * value of weight sources
	
	/**
	 * Creates a neuron with an empty (uninitialized) weights array
	 * <p>
	 * Initialize weights using {@link #initWeights()}, {@link #setWeight(int, float)},
	 * or {@link #addWeight(int, float)}.
	 *
	 * @param weightsSize desired size of weights array, the number of nodes in previous layer (input layer has 0)
	 */
	public Neuron(int weightsSize) {
		weights = new float[weightsSize];
	}
	
	/**
	 * Initialize all weights with random float between -1 and 1
	 */
	public void initWeights() {
		Random random = new Random();
		for (int i = 0; i < weights.length; i++)
			weights[i] = random.nextFloat(-1, 1);
	}
	
	public float getWeight(int idx) {
		return weights[idx];
	}
	
	public float[] getWeights() {
		return weights;
	}
	
	public int getNumWeights() {
		return weights.length;
	}
	
	public void setWeight(int idx, float weight) {
		weights[idx] = weight;
	}
	
	public void addWeight(int index, float value) {
		weights[index] += value;
	}
	
	public float getValue() {
		return value;
	}
	
	public void setValue(float value) {
		this.value = value;
	}
	
	public void resetValue() {
		value = 0;
	}
	
	public void addValue(float value) {
		this.value += value;
	}
}
