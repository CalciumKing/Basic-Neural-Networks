package hybrid_evolution;

import hybrid_evolution.model.Layer;
import hybrid_evolution.model.NeuralNetwork;
import hybrid_evolution.model.Neuron;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * A class representing a trainer that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 *
 * @see #train(float[][], int[], int, float)
 */
public class Trainer {
	private NeuralNetwork[] agents;  // all agents to be used in training session
	
	/**
	 * Initialize trainer and all agents ({@link NeuralNetwork} objects) within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing lengths of each layer
	 *                       ({@code layerLengths.length} will be used to create number of layers)
	 * @throws Exception if {@code agentsPerRound} is not divisible by 4, meaning agents cannot be evenly distributed
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) throws Exception {
		System.out.println("Initializing Agents...");
		
		if (agentsPerRound % 4 != 0)
			throw new Exception(
					"Agent Number Must Be Multiple Of 4 For Even Distribution Of Agents In Each Round, " + agentsPerRound + " Is Not A Multiple Of 4."
			);
		
		agents = new NeuralNetwork[agentsPerRound];
		for (int i = 0; i < agentsPerRound; i++)
			// new network with layer structure as specified in layers variable
			agents[i] = new NeuralNetwork(layerLengths);
	}
	
	/**
	 * Trains all agents ({@link NeuralNetwork} objects) using deep learning,
	 * based on {@code inputs[][]} dataset and desired {@code outputs[]}.
	 * <p>
	 * Repeats training for {@code numRounds}.
	 *
	 * @param inputs    2D array of float inputs representing data to train agents on.
	 * @param outputs   2D array of float outputs representing all the desired outputs.
	 * @param numRounds number of rounds per training session.
	 * @param scale     range to randomly modify weights, should be between 0.0-0.5
	 * @throws InterruptedException if thread error occurs
	 */
	public void train(float[][] inputs, int[] outputs, int numRounds, float scale) throws InterruptedException {
		System.out.println("Training...");
		
		AtomicInteger bestScore = new AtomicInteger(0);
		AtomicReferenceArray<NeuralNetwork> elders = new AtomicReferenceArray<>(agents.length / 4);
		AtomicIntegerArray elderScores = new AtomicIntegerArray(agents.length / 4);
		ArrayList<Thread> threads = new ArrayList<>();
		
		for (int i = 0; i < elders.length(); i++)  // set random default elders
			elders.set(i, agents[randomIndex(elders.length())]);
		
		for (int i = 1; i <= numRounds; i++) {
			AtomicInteger roundBest = new AtomicInteger(0);
			for (NeuralNetwork agent : agents) {
				Thread thread = new Thread(() -> {
					int score = eval(agent, inputs, outputs);  // number of values guessed correctly (0-# Values)
					
					if (score > bestScore.get())
						bestScore.set(score);
					if (score > roundBest.get())
						roundBest.set(score);
					
					saveIfElder(agent, score, elderScores, elders);
				});
				thread.start();
				threads.add(thread);
			}
			
			for (Thread thread : threads)
				thread.join();
			threads.clear();
			
			agents = nextGeneration(scale, elders);
			
			float percent = ((float) roundBest.get() / outputs.length) * 100;
			String formatted = new DecimalFormat("##.##").format(percent);
			System.out.println(i + ": [" + roundBest.get() + "/" + inputs.length + "] (" + formatted + "%)");
		}
		
		float percent = ((float) bestScore.get() / outputs.length) * 100;
		String formatted = new DecimalFormat("#.##").format(percent);
		System.out.println("Best: " + formatted + "%");
	}
	
	/**
	 * Checks if agent is better than any elders, if so,
	 * saves it to {@link AtomicReferenceArray} and score to {@link AtomicIntegerArray}.
	 *
	 * @param agent       current agent being evaluated
	 * @param score       number of values agent has guessed correctly
	 * @param elderScores scores of all the best agents from previous round
	 * @param elders      best agents from previous round
	 */
	private void saveIfElder(NeuralNetwork agent, int score, AtomicIntegerArray elderScores,
	                         AtomicReferenceArray<NeuralNetwork> elders) {
		int length = elderScores.length();
		
		int insertIndex = -1;
		for (int j = 0; j < length; j++) {
			if (score > elderScores.get(j)) {
				insertIndex = j;
				break;
			}
		}
		
		if (insertIndex == -1)
			return;
		
		for (int j = length - 1; j > insertIndex; j--) {
			elderScores.set(j, elderScores.get(j - 1));
			elders.set(j, elders.get(j - 1));
		}
		
		elderScores.set(insertIndex, score);
		elders.set(insertIndex, agent);
	}
	
	/**
	 * Crossbreeds elders to create variations to be used in next generation (crossbred children).
	 * <p>
	 * Creates children with weights of either parent and small
	 * chance of children with weights averaged from both parents (mutation).
	 *
	 * @param elders     agents to be used in crossbreeding
	 * @param childCount number of desired agents
	 * @return array of agents (crossbred children)
	 */
	private NeuralNetwork[] crossbreed(AtomicReferenceArray<NeuralNetwork> elders, int childCount) {
		NeuralNetwork[] crossbredChildren = new NeuralNetwork[childCount];
		
		for (int j = 0; j < childCount; j++) {
			NeuralNetwork parentA = elders.get(randomIndex(elders.length())),
					parentB = elders.get(randomIndex(elders.length()));
			Layer[] parentALayers = parentA.getLayers(),
					parentBLayers = parentB.getLayers(),
					childLayers = new Layer[parentALayers.length];
			
			for (int k = 0; k < parentALayers.length; k++) {
				Layer parentALayer = parentALayers[k],
						parentBLayer = parentBLayers[k],
						childLayer = new Layer(k, parentA, parentALayer.getNumNeurons());
				
				for (int l = 0; l < parentALayer.getNumNeurons(); l++) {
					Neuron parentANeuron = parentALayer.getNeuron(l),
							parentBNeuron = parentBLayer.getNeuron(l),
							childNeuron = new Neuron(parentANeuron.getNumWeights());
					
					for (int n = 0; n < parentANeuron.getNumWeights(); n++) {
						float parentANeuronWeight = parentANeuron.getWeight(n),
								parentBNeuronWeight = parentBNeuron.getWeight(n),
								childWeight;
						
						double random = Math.random();
						if (random < .5f) {
							if (random < .1f)  // mutation average
								childWeight = parentANeuronWeight + parentBNeuronWeight / 2;
							else
								childWeight = parentANeuronWeight;
						} else
							childWeight = parentBNeuronWeight;
						
						childNeuron.setWeight(n, childWeight);
					}
					
					double random = Math.random();
					float value;
					if (random < .5f) {
						if (random < .1f)  // mutation average
							value = parentANeuron.getValue() + parentBNeuron.getValue() / 2;
						else
							value = parentANeuron.getValue();
					} else
						value = parentBNeuron.getValue();
					
					childNeuron.setValue(value);
					childLayer.setNeuron(l, childNeuron);
				}
				
				childLayers[k] = childLayer;
			}
			NeuralNetwork child = new NeuralNetwork(childLayers);
			crossbredChildren[j] = child;
		}
		
		return crossbredChildren;
	}
	
	/**
	 * Funnels all new agents into new array to be used in next generation.
	 *
	 * @param scale  + and - bounds for evolving agents (0.0-0.5)
	 * @param elders best agents from previous round, used to make new agents for next generation
	 * @return new generation of agents to be used in next round
	 */
	private NeuralNetwork[] nextGeneration(float scale, AtomicReferenceArray<NeuralNetwork> elders) {
		NeuralNetwork[] newGeneration = new NeuralNetwork[agents.length];
		
		for (int j = 0; j < elders.length(); j++)
			newGeneration[j] = elders.get(j);
		
		int destPos = elders.length();
		NeuralNetwork[] crossbredChildren = crossbreed(elders, agents.length / 4);
		System.arraycopy(crossbredChildren, 0, newGeneration, destPos, crossbredChildren.length);
		
		NeuralNetwork[] mutantElders = new NeuralNetwork[agents.length / 4];
		for (int j = 0; j < elders.length(); j++)
			mutantElders[j] = elders.get(j).evolve(scale);
		destPos = elders.length() + crossbredChildren.length;
		System.arraycopy(mutantElders, 0, newGeneration, destPos, mutantElders.length);
		
		NeuralNetwork[] newBlood = new NeuralNetwork[agents.length / 4];
		for (int j = 0; j < newBlood.length; j++)
			newBlood[j] = new NeuralNetwork(agents[j].getLayerLengths());
		destPos = elders.length() + crossbredChildren.length + mutantElders.length;
		System.arraycopy(newBlood, 0, newGeneration, destPos, newBlood.length);
		
		return newGeneration;
	}
	
	/**
	 * Returns random number between 0 and upper bound (exclusive)
	 * <p>
	 * Used for generating random indexes
	 * <p>
	 * Uses {@link Random#nextInt(int)}
	 *
	 * @param bound upper bound (exclusive)
	 * @return random int between 0 and bound
	 */
	private int randomIndex(int bound) {
		Random random = new Random();
		return random.nextInt(bound);
	}
	
	/**
	 * Evaluates the accuracy of a single {@link NeuralNetwork}.
	 * <p>
	 * This should be run asynchronously in a {@link Thread} object for fastest results.
	 *
	 * @param network agent to evaluate
	 * @param inputs  2D array of float inputs representing data to train agents on.
	 * @param outputs 2D array of float outputs representing all the desired outputs.
	 * @return number of data sets guessed correctly
	 */
	private int eval(NeuralNetwork network, float[][] inputs, int[] outputs) {
		int score = 0;
		for (int i = 0; i < inputs.length; i++) {
			ArrayList<Float> calculatedOutputs = network.calculate(inputs[i]);
			
			Optional<Float> max = calculatedOutputs.stream().max(Float::compareTo);
			if (max.isEmpty())
				continue;
			
			if (outputs[i] == calculatedOutputs.indexOf(max.get()))
				score++;
		}
		
		return score;
	}
}
