package backpropagation.model;

import java.io.Serializable;
import java.util.Random;

/**
 * A class that represents a single neuron located in a {@link Layer}
 * object, located in a {@link NeuralNetwork} object.
 * <p>
 * Contains weights and a bias - used to calculate a value, used to calculate an input.
 */
public class Neuron implements Serializable {
	private float[] weights;
	private final float[] squaredGradientsSum;
	private float[] weightsChange, gradientChange;
	private float value, bias, error, biasChange;
	
	/**
	 * Creates a neuron with an empty (uninitialized) weights array
	 * <p>
	 * Initialize weights using various weight managing methods.
	 *
	 * @param weightsSize desired size of weights array, the number of nodes in previous layer (input layer has 0)
	 * @see #initWeights(int)
	 * @see #setWeight(int, float)
	 * @see #setWeights(float[])
	 * @see #addWeight(int, float)
	 */
	public Neuron(int weightsSize) {
		weights = new float[weightsSize];
		weightsChange = new float[weightsSize];
		gradientChange = new float[weightsSize];
		squaredGradientsSum = new float[weightsSize];
	}
	
	/**
	 * Initialize all weights and bias with random float between -1 and 1.
	 *
	 * @param seed initial value of the internal state of the pseudorandom
	 *             number generator used to generate weights and biases.
	 * @see Random#Random(long)
	 */
	public Neuron initWeights(int seed) {
		Random random = new Random(seed);
		
		bias = random.nextFloat(-1, 1);
		for (int i = 0; i < weights.length; i++)
			weights[i] = random.nextFloat(-1, 1);
		
		return this;
	}
	
	/**
	 * Calculates value using equation: v = w * v + b of a {@link Layer} object,
	 * then applying it to the sigmoid function.
	 *
	 * @param layer {@link Layer} object containing neurons, used to calculate current neuron's value
	 */
	public void calcValue(Layer layer) {
		value = bias;
		error = 0;
		
		Neuron[] neurons = layer.getNeurons();
		for (int i = 0; i < neurons.length; i++)
			value += neurons[i].value * getWeight(i);
		
		value = sigmoid(value);
	}
	
	/**
	 * Calculates the gradient of the cost function to find the global minimum.
	 *
	 * @param layer {@link Layer} object to get neuron weights from
	 */
	public void calcWeightChange(Layer layer) {
		Neuron[] neurons = layer.getNeurons();
		for (int i = 0; i < neurons.length; i++) {
			float gradient = error * sigmoidDerivative(value);
			gradientChange[i] += gradient;
			weightsChange[i] += gradient * neurons[i].getValue();
		}
		biasChange += error;
	}
	
	/**
	 * Modifies weights based on the square gradient sum formula and {@code learningRate}.
	 * <p>
	 * Modifies biases accordingly, also based on {@code learningRate}.
	 *
	 * @param learningRate small number to modify weights and biases by,
	 *                     used in the square gradient sum formula equation.
	 */
	public void applyWeightChange(float learningRate) {
		for (int i = 0; i < getNumWeights(); i++) {
			float weightLearningRate = learningRate;
			squaredGradientsSum[i] = (float) ((0.9 * squaredGradientsSum[i]) + (0.1 * Math.pow(gradientChange[i], 2)));
			weightLearningRate /= (float) (Math.pow(squaredGradientsSum[i] + 1.0e-8, 0.5));
			
			addWeight(i, weightsChange[i] * weightLearningRate);
		}
		addBias(biasChange * learningRate);
		
		weightsChange = new float[weights.length];
		gradientChange = new float[weights.length];
		biasChange = 0;
	}
	
	/**
	 * Calculates sigmoid function
	 *
	 * @param x value of x in sigmoid function
	 * @return output value of sigmoid function
	 */
	private float sigmoid(float x) {
		return (float) (1 / (1 + Math.exp(-x)));
	}
	
	/**
	 * Calculates derivative of sigmoid function
	 *
	 * @param y value of y in derivative of sigmoid function
	 * @return output value of derivative of sigmoid function
	 */
	private float sigmoidDerivative(float y) {
		return y * (1 - y);
	}
	
	// region Getters/Setters
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
	
	public void setWeights(float[] weights) {
		if (weights.length != this.weights.length)
			throw new RuntimeException(
					"Length of weights parameter does not match number of weights: " + weights.length + " != " + this.weights.length);
		
		this.weights = weights;
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
	
	public void addValue(float value) {
		this.value += value;
	}
	
	
	public float getBias() {
		return bias;
	}
	
	public void setBias(float bias) {
		this.bias = bias;
	}
	
	public void addBias(float bias) {
		this.bias += bias;
	}
	
	
	public float getError() {
		return error;
	}
	
	public void setError(float target) {
		this.error = target - value;
	}
	
	public void addError(float error) {
		this.error += error;
	}
	// endregion
}
