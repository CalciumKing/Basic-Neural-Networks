package hybrid_evolution.model;

/**
 * A class that represents a single layer containing a set number of neurons.
 *
 * @see Neuron
 */
public class Layer {
	private final Neuron[] neurons;  // all neurons in layer
	
	/**
	 * Creates a layer of neurons and initializes all weights within except for input layer.
	 * <p>
	 * Input layer has no previous layer, so no weights are initialized.
	 *
	 * @param layerNum   layer index of current layer, used to initialize weights using previous layer
	 * @param network    {@link NeuralNetwork} object that layer is located in, used to find previous layer
	 * @param numNeurons number of {@link Neuron} objects to include in layer, output layer should
	 *                   be number of possible answers for all input values
	 */
	public Layer(int layerNum, NeuralNetwork network, int numNeurons) {
		neurons = new Neuron[numNeurons];
		
		for (int i = 0; i < numNeurons; i++) {
			if (layerNum == 0) {
				Neuron neuron = new Neuron(0);
				neurons[i] = neuron;
				continue;
			}
			
			int numWeights = network.getLayer(layerNum - 1).getNumNeurons();
			Neuron neuron = new Neuron(numWeights);
			neuron.initWeights();
			neurons[i] = neuron;
		}
	}
	
	public Neuron getNeuron(int idx) {
		return neurons[idx];
	}
	
	public void setNeuron(int idx, Neuron neuron) {
		neurons[idx] = neuron;
	}
	
	public Neuron[] getNeurons() {
		return neurons;
	}
	
	public int getNumNeurons() {
		return neurons.length;
	}
}
