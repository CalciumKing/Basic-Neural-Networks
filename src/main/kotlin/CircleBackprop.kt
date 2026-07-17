import kotlin.random.Random

/**
 * A single-hidden-layer neural network that learns to classify whether a
 * point (x, y) is inside or outside a circle, trained with backpropagation.
 *
 * Architecture: 2 inputs -> [layerLength] hidden neurons -> 1 output.
 * This is the simplest possible network that still needs a hidden layer -
 * one straight-line neuron alone can't carve out a circular boundary, but
 * several straight-line "cuts" combined can approximate one.
 *
 * See DeepCircleNetwork.kt for the same idea generalized to MANY hidden
 * layers - that's what turns "backprop" into "deep learning."
 */

private class BackPropNeuralNetwork(val layerLength: Int, val learningRate: Float) {
    // weights1[i] = the 2 weights (one per input) belonging to hidden neuron i
    val weights1 = Array(layerLength) { DoubleArray(2) { Random.nextDouble(-1.0, 1.0)} }
    val bias1 = DoubleArray(layerLength) { Random.nextDouble(-1.0, 1.0) }

    // weights2[i] = how much hidden neuron i's output influences the final output
    val weights2 = DoubleArray(layerLength) { Random.nextDouble(-1.0, 1.0) }
    var bias2 = Random.nextDouble(-1.0, 1.0)

    /**
     * FORWARD PASS: push a point (x, y) through the network to get a prediction.
     * Returns the hidden layer's activations too, since backprop needs them.
     */
    fun forward(x: Double, y: Double): Pair<DoubleArray, Double> {
        val layer = DoubleArray(layerLength)
        for(i in 0 until weights1.count()) {
            // Each hidden neuron computes its own weighted sum of (x, y) + bias -
            // geometrically, this is one straight line cutting the plane in two.
            val z: Double = weights1[i][0] * x + weights1[i][1] * y + bias1[i]
            // sigmoid squashes z into (0, 1) AND introduces nonlinearity - without
            // this, stacking neurons would be pointless (linear + linear = linear).
            layer[i] = sigmoid(z)
        }

        // Output neuron combines all 6 hidden "votes" into one final weighted sum.
        val zOut: Double = DoubleArray(layerLength) { index -> weights2[index] * layer[index] }.sum() + bias2
        val output = sigmoid(zOut)

        return Pair(layer, output)
    }

    /**
     * BACKPROPAGATION + GRADIENT DESCENT: one training step.
     * Given a point and its correct label, figure out how wrong the
     * prediction was, assign blame backward through the network, then
     * nudge every weight a little bit in the direction that reduces error.
     */
    fun train(x: Double, y: Double, target: Double): Double {
        val (layer, output) = forward(x, y)

        // Step 1: how wrong was the final prediction? (predicted - actual)
        // This simple form falls out of pairing sigmoid with cross-entropy loss.
        val outputError = output - target

        // Step 2: assign blame to each hidden neuron. This is the chain rule -
        // scale the output's error by how much that neuron influenced the output
        // (weights2[index]), then by that neuron's own sensitivity to change
        // (sigmoidDerivative). This is "back"-propagation: error flows backward,
        // opposite the direction data flowed during forward().
        val layerErrors = DoubleArray(layerLength) { index ->
            outputError * weights2[index] * sigmoidDerivative(layer[index])
        }

        // Step 3a: update the output layer's weights (gradient descent).
        // Each weight shifts a little against the error, scaled by learningRate.
        for (i in 0 until weights2.count())
            weights2[i] -= learningRate * outputError * layer[i]
        bias2 -= learningRate * outputError

        // Step 3b: update the hidden layer's weights using the blame (layerErrors)
        // computed in Step 2. Same gradient descent idea, one layer earlier.
        for(i in 0 until weights1.count()) {
            weights1[i][0] -= learningRate * layerErrors[i] * x
            weights1[i][1] -= learningRate * layerErrors[i] * y
            bias1[i] -= learningRate * layerErrors[i]
        }

        return outputError
    }
}

// Draws an ASCII-art snapshot of what the network currently believes the
// boundary looks like - this is the "live" visual that makes learning visible.
private fun printDecision(network: BackPropNeuralNetwork, resolution: Int = 32) {
    println()
    for(row in 0 until resolution) {
        val y: Double = 1 - (2 * row / (resolution.toDouble() - 1))
        var line = ""
        for (col in 0 until resolution) {
            val x: Double = -1 + (2 * col / (resolution.toDouble() - 1))
            val (_, output) = network.forward(x, y)
            line += if (output > .5) '#' else '.'
        }
        println(line)
    }
    println()
}

// Tests the network on fresh random points it wasn't necessarily trained
// on directly, and reports what fraction it classifies correctly.
private fun accuracy(network: BackPropNeuralNetwork, samples: Int = 500): Double {
    var correct = 0
    repeat(samples) {
        val (x, y, target) = randomPoint()
        val (_, output) = network.forward(x, y)
        val predicted = if(output > .5) 1.0 else 0.0
        if (predicted == target)
            correct += 1
    }

    return correct.toDouble() / samples
}

fun main() {
    val network = BackPropNeuralNetwork(6, .5f)
    val generations = 5000
    val checkpoint = 1000

    println("=".repeat(60))
    println("Pre-training (random weights):")
    printDecision(network)
    println("Pre-training accuracy: ${accuracy(network)}")
    println("=".repeat(60))

    // The core training loop: generate a random point, run one full
    // forward + backward (train) step on it, repeat thousands of times.
    // Each call to train() is one tiny nudge toward a better boundary.
    for (gen in 0 until generations) {
        val (x, y, target) = randomPoint()
        network.train(x, y, target)

        if (gen % checkpoint == 0) {
            println("${"=".repeat(25)} Generation $gen ${"=".repeat(25)}")
            printDecision(network)
            println("Accuracy: ${accuracy(network)}")
        }
    }

    println("=".repeat(60))
    println("Final Result:")
    println("Final Accuracy: ${accuracy(network)}")
}
