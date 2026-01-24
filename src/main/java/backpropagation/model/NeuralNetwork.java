package backpropagation.model;

import java.io.Serializable;
import java.util.Random;

/**
 * A class that represents a neural network and all the layers within.
 * Each {@link Layer} contains an array of {@link Neuron} objects.
 * <p>
 * Class can be serialized to save/load the best agent.
 */
public class NeuralNetwork implements Serializable, Cloneable {
	private final Layer[] layers;
	private final int[] layerLengths;
	
	// TODO: create alternative without seed parameter
	/**
	 * Creates a neural network and initializes all layers, neurons, and weights within
	 *
	 * @param layerLengths array containing number of {@link Neuron} objects in each {@link Layer},
	 *                     {@code layerLengths.length} should be total number of layers in network.
	 * @param seed         initial value of the internal state of the pseudorandom number generator used in
	 *                     {@link Neuron} objects inside this network
	 */
	public NeuralNetwork(int[] layerLengths, int seed) {
		layers = new Layer[layerLengths.length];
		this.layerLengths = layerLengths;
		
		for (int i = 0; i < layerLengths.length; i++)
			layers[i] = new Layer(i, this, layerLengths[i], seed);
	}
	
	// TODO: delete if not needed
	/**
	 * Creates a neural network with already initialized {@link Layer} objects.
	 *
	 * @param layers array of already initialized {@link Layer} objects consisting of {@link Neuron} objects.
	 */
	public NeuralNetwork(Layer[] layers) {
		this.layers = layers;
		layerLengths = new int[layers.length];
		
		for (int i = 0; i < layers.length; i++)
			layerLengths[i] = layers[i].getNumNeurons();
	}
	
	/**
	 * Adjust all weights in a network randomly to return a new variation of the current network.
	 *
	 * @param scale how much the weights are changing (+/- bounds for new random evolution),
	 *              should be between 0.0-0.5
	 * @return a clone of the current neural network but with slightly modified ("evolved") weights
	 * @throws CloneNotSupportedException if cloning of current network object fails
	 * @see Layer
	 * @see Neuron
	 */
	public NeuralNetwork evolve(float scale) throws CloneNotSupportedException {
		NeuralNetwork newNetwork = (NeuralNetwork) this.clone();
		Random random = new Random();
		
		for (int i = 1; i < layers.length; i++) {  // skip input layer
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				Neuron newNeuron = getNeuron(i, j);
				for (int k = 0; k < newNeuron.getNumWeights(); k++)
					newNeuron.addWeight(k, random.nextFloat(-scale, scale));
				newNeuron.addBias(random.nextFloat(-scale, scale));
				newNetwork.setNeuron(i, j, newNeuron);
			}
		}
		
		return newNetwork;
	}
	
	/**
	 * Returns network determined values of output layer.
	 * <p>
	 * When return value is compared with definitive answer array, the accuracy of the network can be determined.
	 *
	 * @param inputs values neural network is trained on
	 * @return values of output layer, should be used to compare definitive answer array.
	 * @see Layer
	 * @see Neuron
	 */
	public float[] calcOutputs(float[] inputs) {
		int outputLayerIdx = layers.length - 1;
		float[] outputs = new float[layers[outputLayerIdx].getNumNeurons()];
		
		for (int i = 0; i < layers.length; i++) {
			Neuron[] neurons = layers[i].getNeurons();
			for (int j = 0; j < neurons.length; j++) {
				Neuron neuron = getNeuron(i, j);
				if (i == 0) {
					neuron.setValue(inputs[j]);
					continue;
				}
				
				neuron.calcValue(layers[i - 1]);
				if (i == outputLayerIdx)
					outputs[j] = neuron.getValue();
			}
		}
		
		return outputs;
	}
	
	/**
	 * Apply back propagation process to neural network.
	 *
	 * @param target desired output values
	 * @see Layer
	 * @see Neuron
	 */
	public float[] backProp(float[] target) {
		float[] outputError = new float[target.length];
		
		for (int i = 1; i < layers.length; i++) {
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				Neuron neuron = getNeuron(i, j);
				if (i == layers.length - 1) {
					neuron.setError(target[j]);
					outputError[j] = neuron.getError();
				}
				
				Layer layer = layers[i];
				layer.calcErrors(neuron.getError(), neuron.getWeight(j));
				neuron.calcWeightChange(layer);
			}
		}
		
		return outputError;
	}
	
	/**
	 * Runs {@link Neuron#applyWeightChange(float)} function to all {@link Neuron} objects in network.
	 *
	 * @param learningRate difference to modify weights (0.0-0.5)
	 * @see Layer
	 */
	public void applyWeights(float learningRate) {
		for (int i = 0; i < layers.length; i++)
			for (int j = 0; j < layers[i].getNumNeurons(); j++)
				getNeuron(i, j).applyWeightChange(learningRate);
	}
	
	// region Getters/Setters
	public Layer[] getLayers() {
		return layers;
	}
	
	public int[] getLayerLengths() {
		return layerLengths;
	}
	
	public Layer getLayer(int idx) {
		return layers[idx];
	}
	
	
	public Neuron getNeuron(int layer, int number) {
		return layers[layer].getNeuron(number);
	}
	
	public void setNeuron(int layer, int idx, Neuron neuron) {
		layers[layer].setNeuron(idx, neuron);
	}
	
	
	public void setWeights(float[][][] weights) {
		for (int i = 1; i < layers.length; i++)
			layers[i].setWeights(weights[i - 1]);
	}
	
	public float[][][] getWeights() {
		float[][][] weights = new float[layers.length - 1][][];
		
		for (int i = 1; i < layers.length; i++) {  // ignore input layer
			int numNeurons = layers[i].getNumNeurons();
			weights[i - 1] = new float[numNeurons][];
			for (int j = 0; j < numNeurons; j++)
				weights[i - 1][j] = getNeuron(i, j).getWeights();
		}
		
		return weights;
	}
	
	
	public void setBiases(float[][] biases) {
		for (int i = 1; i < layerLengths.length; i++)
			for (int j = 0; j < layerLengths[i]; j++)
				getNeuron(i, j).setBias(biases[i - 1][j]);
	}
	
	public float[][] getBiases() {
		float[][] biases = new float[layers.length - 1][];
		
		for (int i = 1; i < layers.length; i++) {  // ignore input layer
			int numNeurons = layers[i].getNumNeurons();
			biases[i - 1] = new float[numNeurons];
			for (int j = 0; j < numNeurons; j++)
				biases[i - 1][j] = getNeuron(i, j).getBias();
		}
		
		return biases;
	}
	// endregion
}
