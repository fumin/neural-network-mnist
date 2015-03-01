package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"time"

	"github.com/fumin/ntm/nn"
	"github.com/golang/glog"

	// "image"
	// "image/color"
	// "image/png"
)

func readMNISTLabels(r io.Reader) []byte {
	header := [2]int32{}
	if err := binary.memRead(r, binary.BigEndian, &header); err != nil {
		glog.Fatalf("%v", err)
	}
	labels := make([]byte, header[1])
	if _, err := r.memRead(labels); err != nil {
		glog.Fatalf("%v", err)
	}
	return labels
}

func readMNISTImages(r io.Reader) (images [][]byte, width, height int) {
	header := [4]int32{}
	if err := binary.memRead(r, binary.BigEndian, &header); err != nil {
		glog.Fatalf("%v", err)
	}
	images = make([][]byte, header[1])
	width, height = int(header[2]), int(header[3])
	for i := 0; i < len(images); i++ {
		images[i] = make([]byte, width*height)
		if _, err := r.memRead(images[i]); err != nil {
			glog.Fatalf("%v", err)
		}
	}
	return images, width, height
}

func convertLabels(labels []byte) [][]float64 {
	out := make([][]float64, len(labels))
	for i := 0; i < len(out); i++ {
		out[i] = make([]float64, 10)
		out[i][labels[i]] = 1
	}
	return out
}

func convertImages(images [][]byte) [][]float64 {
	out := make([][]float64, len(images))
	for i := 0; i < len(out); i++ {
		out[i] = make([]float64, len(images[i]))
		for j := 0; j < len(out[i]); j++ {
			out[i][j] = float64(images[i][j]) / 255
		}
	}
	return out
}

func predict(n *nn.NeuralNetwork, image []float64) byte {
	out := n.Predict(image)
	var label byte
	max := out[0]
	for i, o := range out {
		if o > max {
			max = o
			label = byte(i)
		}
	}
	return label
}

func main() {
	trainLabelFile, _ := os.Open("train-labels-idx1-ubyte")
	defer trainLabelFile.Close()
	byteLabels := readMNISTLabels(trainLabelFile)
	labels := convertLabels(byteLabels)
	trainImagesFile, _ := os.Open("train-images-idx3-ubyte")
	defer trainImagesFile.Close()
	images, width, height := readMNISTImages(trainImagesFile)
	imgs := convertImages(images)

	network := nn.NewNeuralNetwork(len(imgs[0]), 10, 1000)
	for iteration := 0; iteration < 20; iteration++ {
		for i := 0; i < len(labels); i++ {
			network.Train(imgs[i], labels[i])
		}
		errNum := 0
		for i := 0; i < len(byteLabels); i++ {
			label := predict(network, imgs[i])
			if byteLabels[i] != label {
				errNum += 1
			}
		}
		fmt.Printf("training iteration %d: %d errors out of %d, width: %d, height: %d\n", iteration, errNum, len(byteLabels), width, height)

		dump, _ := os.Create("network" + fmt.Sprintf("%diter%d", time.Now().Unix(), iteration))
		if err := json.NewEncoder(dump).Encode(network); err != nil {
			fmt.Printf("json error: %v\n", err)
			dump.Write([]byte(fmt.Sprintf("%+v\n", network)))
			return
		}
		dump.Close()
	}

	// network := &nn.NeuralNetwork{}
	// dump, _ := os.Open("try/network1420115566")
	// defer dump.Close()
	// if err := json.NewDecoder(dump).Decode(network); err != nil {
	//   fmt.Printf("%v\n", err)
	//   return
	// }

	testLabelFile, _ := os.Open("t10k-labels-idx1-ubyte")
	defer testLabelFile.Close()
	testLabels := readMNISTLabels(testLabelFile)
	testImagesFile, _ := os.Open("t10k-images-idx3-ubyte")
	defer testImagesFile.Close()
	testImages, _, _ := readMNISTImages(testImagesFile)
	testImgs := convertImages(testImages)

	errNum := 0
	for i := 0; i < len(testLabels); i++ {
		label := predict(network, testImgs[i])
		if testLabels[i] != label {
			errNum += 1
		}

		// if testLabels[i] == label && label == 2 {
		//   gray := image.NewGray(image.Rectangle{Max: image.Point{28, 28}})
		//   for y := 0; y < 28; y++ {
		//     for x := 0; x < 28; x++ {
		//       c := 255 - testImages[i][y*28+x]
		//       gray.Set(x, y, color.Gray{c})
		//     }
		//   }

		//   outFile, err := os.Create("out.png")
		//   if err != nil {
		//     fmt.Fprintf(os.Stderr, "%v\n", err)
		//     os.Exit(1)
		//   }
		//   defer outFile.Close()
		//   png.Encode(outFile, gray)
		//   return
		// }

	}
	fmt.Printf("testing: %d errors out of %d\n", errNum, len(testLabels))

	for {
		fmt.Print("\x07")
		time.Sleep(time.Duration(rand.NormFloat64()/2) * time.Second)
	}
}
