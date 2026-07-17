import kotlin.random.Random

/**
 * DEEP LEARNING vs the single-hidden-layer version (CircleBackprop.kt):
 *
 * "Deep learning" just means MULTIPLE hidden layers stacked, instead of one.
 * Backpropagation doesn't change conceptually - it's the same chain-rule
 * idea, just repeated once per layer, walking backward from the output
 * all the way to the input.
 *
 * This network takes a list of layer sizes, e.g. [2, 6, 6, 1] means:
 *   2 inputs -> 6 hidden neurons -> 6 hidden neurons -> 1 output
 * You can make it "deeper" just by adding more numbers to that list.
 */

private class DeepNeuralNetwork(sizes: IntArray, private val learningRate: Double) {
    // weights[layer][neuronInThisLayer][neuronInPreviousLayer]
    private val weights: MutableList<Array<DoubleArray>> = mutableListOf()
    // biases[layer][neuronInThisLayer]
    private val biases: MutableList<DoubleArray> = mutableListOf()

    init {
        for (i in 0 until sizes.size - 1) {
            val fanIn = sizes[i]
            val fanOut = sizes[i + 1]
            weights.add(Array(fanOut) { DoubleArray(fanIn) { Random.nextDouble(-1.0, 1.0) } })
            biases.add(DoubleArray(fanOut) { Random.nextDouble(-1.0, 1.0) })
        }
    }

    /**
     * Runs the input all the way through every layer.
     * Returns the activation (output) of EVERY layer, including the input
     * itself at index 0 - we need all of these later during backprop.
     */
    fun forward(inputs: DoubleArray): List<DoubleArray> {
        val activations = mutableListOf(inputs)
        var a = inputs

        for (layerIndex in weights.indices) {
            val w = weights[layerIndex]
            val b = biases[layerIndex]
            val newA = DoubleArray(w.size)
            for (j in w.indices) {
                var z = b[j]
                for (k in a.indices)
                    z += w[j][k] * a[k]
                newA[j] = sigmoid(z)
            }
            a = newA
            activations.add(a)
        }

        return activations
    }

    /**
     * One full training step: forward pass, then backpropagate the error
     * through every layer, from the output back to the first hidden layer.
     */
    fun train(inputs: DoubleArray, target: Double) {
        val activations = forward(inputs)
        val output = activations.last()[0]
        val outputError = output - target

        // deltas[layer] = error signal for every neuron in that layer
        val deltas = arrayOfNulls<DoubleArray>(weights.size)

        // Step 1: error at the very last layer (the output)
        deltas[deltas.size - 1] = doubleArrayOf(outputError * sigmoidDerivative(activations.last()[0]))

        // Step 2: walk BACKWARD through every hidden layer, propagating
        // error using the SAME chain-rule idea each time - this loop is
        // the only real difference from the single-hidden-layer version.
        for (layerIndex in weights.size - 2 downTo 0) {
            val nextWeights = weights[layerIndex + 1]
            val nextDelta = deltas[layerIndex + 1]!!
            val currentActivation = activations[layerIndex + 1]

            val delta = DoubleArray(currentActivation.size)
            for (j in currentActivation.indices) {
                var errorSum = 0.0
                for (k in nextDelta.indices)
                    errorSum += nextDelta[k] * nextWeights[k][j]
                delta[j] = errorSum * sigmoidDerivative(currentActivation[j])
            }
            deltas[layerIndex] = delta
        }

        // Step 3: gradient descent - nudge every weight and bias in every
        // layer, using that layer's delta and the activation feeding INTO it.
        for (layerIndex in weights.indices) {
            val prevActivation = activations[layerIndex]
            val delta = deltas[layerIndex]!!
            for (j in weights[layerIndex].indices) {
                for (k in prevActivation.indices)
                    weights[layerIndex][j][k] -= learningRate * delta[j] * prevActivation[k]
                biases[layerIndex][j] -= learningRate * delta[j]
            }
        }
    }
}

private fun accuracy(network: DeepNeuralNetwork, samples: Int = 1000): Double {
    var correct = 0
    repeat(samples) {
        val (x, y, target) = randomPoint()
        val output = network.forward(doubleArrayOf(x, y)).last()[0]
        val predicted = if (output > 0.5) 1.0 else 0.0
        if (predicted == target) correct++
    }
    return correct.toDouble() / samples
}

private fun printDecision(network: DeepNeuralNetwork, resolution: Int = 32) {
    println()
    for (row in 0 until resolution) {
        val y = 1 - (2 * row / (resolution.toDouble() - 1))
        val line = StringBuilder()
        for (col in 0 until resolution) {
            val x = -1 + (2 * col / (resolution.toDouble() - 1))
            val output = network.forward(doubleArrayOf(x, y)).last()[0]
            line.append(if (output > 0.5) '#' else '.')
        }
        println(line)
    }
    println()
}

fun main() {
    val network = DeepNeuralNetwork(listOf(2, 6, 6, 1).toIntArray(), .5)
    val epochs = 20000
    val checkpoint = 4000

    println("=".repeat(60))
    println("Pre-training (random weights):")
    printDecision(network)
    println("Pre-training accuracy: ${accuracy(network)}")
    println("=".repeat(60))

    for (epoch in 1..epochs) {
        val (x, y, target) = randomPoint()
        network.train(doubleArrayOf(x, y), target)

        if (epoch % checkpoint == 0) {
            println("${"=".repeat(20)} Epoch $epoch ${"=".repeat(20)}")
            printDecision(network)
            println("Accuracy: ${accuracy(network)}")
        }
    }

    println("=".repeat(60))
    println("Final Result:")
    println("Final Accuracy: ${accuracy(network, 2000)}")
}
