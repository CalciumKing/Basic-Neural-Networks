package backpropagation;

import backpropagation.model.NeuralNetwork;

import java.text.DecimalFormat;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A class representing a trainer object that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 *
 * @see #getBestScore()
 * @see #train(float[][], float[][], int[], float, int)
 */
public class Trainer {
	private final AtomicReference<NeuralNetwork> bestAgent;
	private final NeuralNetwork[] agents;
	private final AtomicLong bestScore;
	
	/**
	 * Initialize trainer and all agents ({@link NeuralNetwork} objects) within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing lengths of each layer
	 *                       ({@code layerLengths.length} will be used to create number of layers)
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) {
		bestAgent = new AtomicReference<>(new NeuralNetwork(layerLengths, 123));
		bestScore = new AtomicLong(0);
		agents = new NeuralNetwork[agentsPerRound];
		for (int i = 0; i < agentsPerRound; i++)
			agents[i] = new NeuralNetwork(layerLengths, 123);
	}
	
	/**
	 * Trains a single {@link NeuralNetwork} agent using the gradient decent algorithm with back propagation.
	 *
	 * @param agent        {@link NeuralNetwork} object to train with
	 * @param inputs       values neural network is trained on
	 * @param targets      calculated values of output layer
	 * @param outputs      desired values of the output layer
	 * @param learningRate difference to modify weights (0.0-0.5)
	 * @return number of data points the agent got correct
	 */
	private float trainAgent(NeuralNetwork agent, float[][] inputs, float[][] targets,
	                         int[] outputs, float learningRate) {
		float[] MSE = new float[targets[0].length];
		int score = 0;
		
		for (int i = 0; i < inputs.length; i++) {
			float[] calcOutputs = agent.calcOutputs(inputs[i]);
			
			int maxIndex = 0;
			for (int j = 0; j < calcOutputs.length; j++) {
				if (calcOutputs[j] > calcOutputs[maxIndex]) {
					calcOutputs[maxIndex] = calcOutputs[j];
					maxIndex = j;
				}
			}
			if (maxIndex == outputs[i])
				score++;
			
			float[] outputErrors = agent.backProp(targets[i]);
			for (int j = 0; j < outputErrors.length; j++)
				MSE[j] += (float) Math.pow(outputErrors[j], 2);
		}
		agent.applyWeights(learningRate);
		
		for (int i = 0; i < MSE.length; i++)
			MSE[i] /= inputs.length;
		
		return score;
	}
	
	/**
	 * Trains all {@link NeuralNetwork} objects within current trainer object.
	 *
	 * @param inputs        values neural network is trained on
	 * @param targets       calculated values of output layer
	 * @param outputs       desired values of the output layer
	 * @param learningRate  difference to modify weights (0.0-0.5)
	 * @param generationNum generation number of current training session
	 * @throws Exception if file logging fails
	 */
	public void train(float[][] inputs, float[][] targets, int[] outputs,
	                  float learningRate, int generationNum) throws Exception {
		float[] scores = new float[agents.length];
		for (int i = 0; i < agents.length; i++)
			scores[i] = trainAgent(agents[i], inputs, targets, outputs, learningRate);
		
		int bestIndex = 0;
		for (int i = 0; i < scores.length; i++)
			if (scores[i] > scores[bestIndex])
				bestIndex = i;
		
		float bestRoundScore = scores[bestIndex];
		float percent = bestRoundScore / inputs.length * 100;
		String formatted = new DecimalFormat("###.##").format(percent);
		
		System.out.println(
				"Generation: " + generationNum + " | Best: [" + bestRoundScore + "/" + inputs.length + "] (" + formatted + "%)");
		
		if (bestRoundScore > bestScore.get()) {
			bestScore.set((long) bestRoundScore);
			bestAgent.set(agents[bestIndex]);
		}
		
		NeuralNetwork bestRoundAgent = agents[bestIndex];
		for (int i = 0; i < agents.length; i++) {
			if (i == bestIndex) {
				agents[i] = bestRoundAgent;
				continue;
			}
			
			agents[i] = bestRoundAgent.evolve(learningRate);
		}
	}
	
	public float getBestScore() {
		return bestScore.get();
	}
}
