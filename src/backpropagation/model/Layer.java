package backpropagation.model;

import java.io.Serializable;

/**
 * A class that represents a single layer containing a set number of
 * {@link Neuron} objects.
 * <p>
 * A {@link NeuralNetwork} object contains at least 2 layers.
 */
public class Layer implements Serializable {
	private final Neuron[] neurons;
	
	/**
	 * Creates a layer of neurons and initializes all weights within except for input layer.
	 * <p>
	 * Input layer has no previous layer, so no weights are initialized.
	 *
	 * @param layerNum   layer index of current layer, used to initialize weights using previous layer
	 * @param network    {@link NeuralNetwork} object that layer is located in, used to find previous layer
	 * @param numNeurons number of {@link Neuron} objects to include in layer, output layer should
	 *                   be number of possible answers for all input values
	 * @param seed       initial value of the internal state of the pseudorandom number generator used in
	 *                   {@link Neuron} objects inside this network
	 */
	public Layer(int layerNum, NeuralNetwork network, int numNeurons, int seed) {
		neurons = new Neuron[numNeurons];
		
		for (int i = 0; i < numNeurons; i++) {
			if (layerNum == 0) {
				Neuron neuron = new Neuron(0);
				neurons[i] = neuron;
				continue;
			}
			
			int numWeights = network.getLayer(layerNum - 1).getNumNeurons();
			neurons[i] = new Neuron(numWeights).initWeights(seed);
		}
	}
	
	/**
	 * Calculates errors in current object based on {@link Neuron} objects within.
	 *
	 * @param error  difference between layer value and target value;
	 *               smaller error means better accuracy;
	 *               calculated using {@link Neuron#getError()}
	 * @param weight a singular weight value in next layer;
	 *               calculated using {@link Neuron#getWeight(int)}
	 */
	public void calcErrors(float error, float weight) {
		for (Neuron neuron : getNeurons())
			neuron.addError(error * weight);
	}
	
	// region Getters/Setters
	public Neuron[] getNeurons() {
		return neurons;
	}
	
	public int getNumNeurons() {
		return neurons.length;
	}
	
	public Neuron getNeuron(int idx) {
		return neurons[idx];
	}
	
	public void setNeuron(int idx, Neuron neuron) {
		neurons[idx] = neuron;
	}
	
	public void setWeights(float[][] weights) {
		for (int i = 0; i < getNumNeurons(); i++)
			getNeuron(i).setWeights(weights[i]);
	}
	// endregion
}
