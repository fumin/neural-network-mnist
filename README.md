Feedforward neural networks on MNIST
-----
This is preliminary experiment for the implementation of Neural Turing Machines https://github.com/fumin/ntm/ . The goal is to come up with a working neural network to plug into a NTM as a controller.

Here are the results achieved with no momentum, plain SGD with learning rate 0.1, sigmoid non-linearities:
* 100 hidden units, errors -> training: 657/60000, testing: 389/10000
* 1000 hidden units, 20 iterations -> training: 269/60000, testing: 225/10000, training time took 2 hours on a Macbook Pro (Mid 2012) 2.3 GHz Intel Core i7.

To try out this implementation on real handwritings, run
* `cd mnist/try`
* `go run try.go =img=eight.png`
* The network correctly predicts that the number written is eight.

The images directly under the folder "mnist/try" are handwritings of mine, the ones under "mnist/try/mariusz" are Mariusz's, the ones under "mnist/try/yenting" are YenTing's.
