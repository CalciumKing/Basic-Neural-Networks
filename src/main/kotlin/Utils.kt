import kotlin.math.exp
import kotlin.random.Random

const val RADIUS = .6

// Squashes any real number into (0, 1) - lets us read the output as a
// confidence/probability, and gives the network its nonlinearity.
fun sigmoid(z: Double): Double = 1 / (1 + exp(-z))

// Derivative of sigmoid, expressed in terms of its OWN output (a = sigmoid(z)).
// This is the piece backprop needs to know "how sensitive" a neuron is.
fun sigmoidDerivative(a: Double): Double = a * (1 - a)

fun randomPoint(): Triple<Double, Double, Double> {
    val x = Random.nextDouble(-1.0, 1.0)
    val y = Random.nextDouble(-1.0, 1.0)
    return Triple(x, y, label(x, y))
}

fun label(x: Double, y: Double): Double {
    return if ((x * x + y * y) < RADIUS * RADIUS) 1.0 else 0.0
}
