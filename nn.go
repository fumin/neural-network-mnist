package nn

import (
	"math"
	"math/rand"
)

type NeuralNetwork struct {
	Wih1 [][]float64 // weight matrix connecting inputs to the 1st hidden layer.
	B1h  []float64   // bias term of 1st hidden layer.

	// rmsprop
	ni      [][]float64
	gi      [][]float64
	deltai  [][]float64
	nib     []float64
	gib     []float64
	deltaib []float64

	Whny [][]float64 // weights of output layer.
	By   []float64   //bias term of output layer.
}

func NewNeuralNetwork(inSize, outSize, hiddenSize int) *NeuralNetwork {
	n := NeuralNetwork{}
	n.Wih1 = make([][]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		n.Wih1[i] = make([]float64, inSize)
		for j := 0; j < inSize; j++ {
			n.Wih1[i][j] = rand.NormFloat64()
		}
	}
	n.B1h = make([]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		n.B1h[i] = rand.NormFloat64()
	}

	n.Whny = make([][]float64, outSize)
	for i := 0; i < outSize; i++ {
		n.Whny[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			n.Whny[i][j] = rand.NormFloat64()
		}
	}
	n.By = make([]float64, outSize)
	for i := 0; i < outSize; i++ {
		n.By[i] = rand.NormFloat64()
	}

	// rmsprop
	n.ni = make([][]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		n.ni[i] = make([]float64, inSize)
	}
	n.gi = make([][]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		n.gi[i] = make([]float64, inSize)
	}
	n.deltai = make([][]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		n.deltai[i] = make([]float64, inSize)
	}
	n.nib = make([]float64, hiddenSize)
	n.gib = make([]float64, hiddenSize)
	n.deltaib = make([]float64, hiddenSize)

	return &n
}

func (n *NeuralNetwork) forward1stLayer(in []float64) []float64 {
	h1t := make([]float64, len(n.B1h))
	for i := 0; i < len(h1t); i++ {
		for j := 0; j < len(in); j++ {
			h1t[i] += n.Wih1[i][j] * in[j]
		}
	}
	for i := 0; i < len(h1t); i++ {
		h1t[i] += n.B1h[i]
	}
	// Rectified linear units
	// for i := 0; i < len(h1t); i++ {
	//   if h1t[i] < 0 {
	//     h1t[i] = 0
	//   }
	// }

	for i := 0; i < len(h1t); i++ {
		h1t[i] = Sigmoid(h1t[i])
	}

	return h1t
}

func (n *NeuralNetwork) forwardY(in []float64) []float64 {
	yt := make([]float64, len(n.By))
	for i := 0; i < len(yt); i++ {
		for j := 0; j < len(in); j++ {
			yt[i] += n.Whny[i][j] * in[j]
		}
	}
	for i := 0; i < len(yt); i++ {
		yt[i] += n.By[i]
	}

	// Rectified sigmoid
	// for i := 0; i < len(yt); i++ {
	//   if yt[i] < 0 {
	//     yt[i] = 0
	//   }
	//   if yt[i] > 1 {
	//     yt[i] = 1
	//   }
	// }

	for i := 0; i < len(yt); i++ {
		yt[i] = Sigmoid(yt[i])
	}
	return yt
}

func (n *NeuralNetwork) Train(in, out []float64) {
	// Forward
	h1t := n.forward1stLayer(in)
	yt := n.forwardY(h1t)

	// Backwards
	ytDelta := make([]float64, len(out))
	for i := 0; i < len(ytDelta); i++ {
		ytDelta[i] = out[i] - yt[i]
	}
	h1tDelta := make([]float64, len(h1t))
	for i := 0; i < len(h1t); i++ {
		// if h1t[i] < 0 { // Rectified linear unit did not fire
		//   continue
		// }
		for j := 0; j < len(ytDelta); j++ {
			h1tDelta[i] += n.Whny[j][i] * ytDelta[j] * h1t[i] * (1 - h1t[i])
		}
	}

	rate := 0.1
	for i := 0; i < len(out); i++ {
		for j := 0; j < len(h1t); j++ {
			n.Whny[i][j] += rate * clamp(ytDelta[i]*h1t[j], 100)
		}
		n.By[i] += rate * clamp(ytDelta[i], 100)
	}

	for i := 0; i < len(h1t); i++ {
		for j := 0; j < len(in); j++ {
			// Stochastic Gradient Descent with momentum, works by decreasing error from
			// 652/10000 to 600/10000 (rate 0.1, mu 0.2)
			// diff := rate * clamp(h1tDelta[i] * in[j], 100) + 0.2 * n.ni[i][j]
			// n.Wih1[i][j] += diff
			// n.ni[i][j] = diff

			// Standard Rmsprop, disaster
			// epsilon := clamp(h1tDelta[i] * in[j], 10)
			// if epsilon > 0.00001 {
			//   n.ni[i][j] = 0.9 * n.ni[i][j] + (1-0.9) * epsilon * epsilon
			//   n.Wih1[i][j] += rate / math.Sqrt(n.ni[i][j]) * epsilon
			// } else {
			//   n.Wih1[i][j] += rate * clamp(h1tDelta[i] * in[j], 100)
			// }

			// Rmsprop as in Generating Sequences With Recurrent Neural Networks, Alex Graves, disaster
			// n.ni[i][j] = 0.95 * n.ni[i][j] + (1-0.95) * epsilon * epsilon
			// n.gi[i][j] = 0.95 * n.gi[i][j] + (1-0.95) * epsilon
			// n.deltai[i][j] = 0.9 * n.deltai[i][j] - 0.0001*epsilon/math.Sqrt(n.ni[i][j] - n.gi[i][j]*n.gi[i][j] + 0.0001)
			// n.deltai[i][j] = clamp(n.deltai[i][j], 10)
			// n.Wih1[i][j] += n.deltai[i][j]

			n.Wih1[i][j] += rate * clamp(h1tDelta[i]*in[j], 100)
		}
		// diff := rate * clamp(h1tDelta[i], 100) + 0.2 * n.nib[i]
		// n.B1h[i] += diff
		// n.nib[i] = diff

		// epsilon := clamp(h1tDelta[i], 10)
		// if epsilon > 0.00001 {
		//   n.nib[i] = 0.9 * n.nib[i] + (1-0.9) * epsilon * epsilon
		//   n.B1h[i] += rate / math.Sqrt(n.nib[i]) * epsilon
		// } else {
		//   n.B1h[i] += rate * clamp(h1tDelta[i], 100)
		// }

		// n.nib[i] = 0.95 * n.nib[i] + (1-0.95) * epsilon * epsilon
		// n.gib[i] = 0.95 * n.gib[i] + (1-0.95) * epsilon
		// n.deltaib[i] = 0.9 * n.deltaib[i] - 0.0001*epsilon/math.Sqrt(n.nib[i] - n.gib[i]*n.gib[i] + 0.0001)
		// n.deltaib[i] = clamp(n.deltaib[i], 10)
		// n.B1h[i] += n.deltaib[i]

		n.B1h[i] += rate * clamp(h1tDelta[i], 100)
	}
}

func (n *NeuralNetwork) Predict(in []float64) []float64 {
	return n.forwardY(n.forward1stLayer(in))
}

func clamp(f float64, max float64) float64 {
	if f > max {
		return max
	}
	if f < -max {
		return -max
	}
	return f
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}
