package hybrid_evolution.model;

import java.util.ArrayList;
import java.util.Random;

/**
 * A class that represents a neural network and all the layers within.
 *
 * @see Layer
 */
public class NeuralNetwork {
	private final Layer[] layers;  // network consisting of layers
	private final int[] layerLengths;  // length of all layers in network
	
	/**
	 * Creates a neural network and initializes all layers, neurons, and weights within
	 *
	 * @param layerLengths array containing number of neurons in each layer,
	 *                     {@code layerLengths.length} should be number of layers in network
	 * @see Layer
	 * @see Neuron
	 */
	public NeuralNetwork(int[] layerLengths) {
		layers = new Layer[layerLengths.length];
		this.layerLengths = layerLengths;
		
		for (int i = 0; i < layerLengths.length; i++) {
			Layer layer = new Layer(i, this, layerLengths[i]);
			layers[i] = layer;
		}
	}
	
	/**
	 * Creates a neural network with already initialized layers
	 *
	 * @param layers array of already initialized layers of neurons
	 * @see Layer
	 * @see Neuron
	 */
	public NeuralNetwork(Layer[] layers) {
		this.layers = layers;
		layerLengths = new int[layers.length];
		
		for (int i = 0; i < layers.length; i++)
			layerLengths[i] = layers[i].getNumNeurons();
	}
	
	public Layer getLayer(int idx) {
		return layers[idx];
	}
	
	public Layer[] getLayers() {
		return layers;
	}
	
	public int[] getLayerLengths() {
		return layerLengths;
	}
	
	/**
	 * Adjust all weights in a network randomly to return a new variation of the network
	 *
	 * @param scale how much the weights are changing (+ and - bounds for new random evolution)
	 * @return a new neural network with modified ("evolved") weights
	 * @see Layer
	 * @see Neuron
	 */
	public NeuralNetwork evolve(float scale) {
		NeuralNetwork newNetwork = this;
		Random random = new Random();
		
		for (int i = 1; i < layers.length; i++) {  // skip input layer
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				for (int k = 0; k < layers[i].getNeuron(j).getNumWeights(); k++) {
					float randFloat = random.nextFloat(-scale, scale);
					newNetwork.getNeuron(i, j).addWeight(k, randFloat);
				}
			}
		}
		
		return newNetwork;
	}
	
	/**
	 * Returns network determined values of output layer, essentially the "run" function.
	 * <p>
	 * When return value is compared with definitive answer array, the accuracy of the network can be determined.
	 *
	 * @param inputs values neural network is trained on
	 * @return values of output layer, should be used to compare definitive answer array.
	 * @see Layer
	 * @see Neuron
	 */
	public ArrayList<Float> calculate(float[] inputs) {
		for (Layer layer : layers)
			for (Neuron neuron : layer.getNeurons())
				neuron.resetValue();  // resets all neuron values to 0
		
		for (int i = 0; i < layers.length; i++) {
			Neuron[] neurons = layers[i].getNeurons();
			for (int j = 0; j < neurons.length; j++) {
				if (i == 0) {
					getNeuron(i, j).setValue(inputs[j]);
					continue;
				}
				
				Neuron neuron = getNeuron(i, j),
						prevNeuron = getNeuron(i - 1, j);
				for (int k = 0; k < neuron.getNumWeights(); k++)
					neuron.addValue(prevNeuron.getValue() * neuron.getWeight(k));
			}
		}
		
		ArrayList<Float> outputs = new ArrayList<>();
		Layer outputLayer = layers[layers.length - 1];
		for (Neuron neuron : outputLayer.getNeurons())
			outputs.add(neuron.getValue());
		
		return outputs;
	}
	
	public Neuron getNeuron(int layer, int number) {
		return layers[layer].getNeuron(number);
	}
}
