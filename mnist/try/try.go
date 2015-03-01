package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"os"
	"sort"

	"github.com/fumin/ntm/nn"
	"github.com/fumin/ntm/nn/mnist/try/resize"
)

const (
	mnistW = 28 // MNIST images are of width and height 28 pixels
)

var (
	imgFilename = flag.String("img", "", "image file name")
)

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
	fmt.Printf("%v\n", out)
	return label
}

func drawIn(in []float64) string {
	out := ""
	for y := 0; y < mnistW; y++ {
		for x := 0; x < mnistW; x++ {
			if in[y*mnistW+x] > 0.5 {
				out += "*"
			} else {
				out += "_"
			}
		}
		out += "\n"
	}
	return out
}

type intSlice []int

func (a intSlice) Len() int           { return len(a) }
func (a intSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a intSlice) Less(i, j int) bool { return a[i] < a[j] }

// backgroundPixel determines the threshold value of the background of a gray scale image.
func backgroundPixel(gray image.Image) uint8 {
	return 128
	pixels := make([]int, 0)
	for x := gray.Bounds().Min.X; x < gray.Bounds().Max.X; x++ {
		for y := gray.Bounds().Min.Y; y < gray.Bounds().Max.Y; y++ {
			pixels = append(pixels, int(gray.At(x, y).(color.Gray).Y))
		}
	}
	sort.Sort(intSlice(pixels))
	th := len(pixels) * 7 / 8
	return uint8(pixels[th])
}

func main() {
	flag.Parse()
	f, err := os.Open(*imgFilename)
	if err != nil {
		flag.Usage()
		os.Exit(1)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	resized := resize.Resize(img, img.Bounds(), mnistW, mnistW)
	grayResized := image.NewGray(image.Rectangle{Max: image.Point{mnistW, mnistW}})
	for x := resized.Bounds().Min.X; x < resized.Bounds().Max.X; x++ {
		for y := resized.Bounds().Min.Y; y < resized.Bounds().Max.Y; y++ {
			grayColor := color.GrayModel.Convert(resized.At(x, y)).(color.Gray)
			grayColor.Y = 255 - grayColor.Y
			grayResized.Set(x-resized.Bounds().Min.X, y-resized.Bounds().Min.Y, grayColor)
		}
	}
	// Convert to MNIST bilevel format
	avg := backgroundPixel(grayResized)
	gray := image.NewGray(image.Rectangle{Max: image.Point{mnistW, mnistW}})
	for x := 0; x < mnistW; x++ {
		for y := 0; y < mnistW; y++ {
			grayColor := grayResized.At(x, y).(color.Gray)
			if grayColor.Y > avg {
				grayColor.Y = 255
			} else {
				grayColor.Y = 0
			}
			gray.Set(x, y, grayColor)
		}
	}

	outFile, err := os.Create("out.png")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	defer outFile.Close()
	png.Encode(outFile, gray)

	network := &nn.NeuralNetwork{}
	dump, _ := os.Open("network1000hidden20iters")
	defer dump.Close()
	if err := json.NewDecoder(dump).Decode(network); err != nil {
		fmt.Printf("%v\n", err)
		os.Exit(1)
	}

	in := make([]float64, mnistW*mnistW)
	for y := 0; y < mnistW; y++ {
		for x := 0; x < mnistW; x++ {
			in[y*mnistW+x] = float64(gray.At(x, y).(color.Gray).Y) / 255
		}
	}
	res := predict(network, in)
	fmt.Printf("%s\n", drawIn(in))
	fmt.Printf("%d\n", res)
}
