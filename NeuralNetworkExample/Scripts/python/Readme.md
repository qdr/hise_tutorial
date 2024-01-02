# Python scripts & data models

This folder contains a few helper scripts, mainly for improving the workflow when using PyTorch. I haven't used Tensorflow except for loading in the trained model from `model_tf.json`.

- `model.py`: A simple PyTorch model for approximating functions
- `dataset.py`: A simple data generator that creates input values and puts them through a function
- `weights.pth`: The trained weights for the function `y = tanh(15.0 * x)` (a simple overdrive)
- `exporter.py`: An exporter script that creates a JSON file containing the weights and the model layout
- `model.json`: The exported script that coalesces the model and the trained weights
- `model_tf.json`: Another JSON definition from a model that was trained with Tensorflow (taken from [here](https://github.com/jatinchowdhury18/RTNeural-example/blob/main/neural_net_weights.json))

# PyTorch workflow

I've toyed around with Pytorch and the TorchStudio program in particular and found this to be the easiest way to get started as it gives you almost a no-code solution. So if your Python skills are as dusted as mine, it will save you hours of googling "how to access an array element in Python" until you will get to the juicy stuff. In the following we're doing a complete roundtrip from creating a dataset over building and training a model to exporting it and loading it into HISE. 

Usually the first thing you'll do when starting out with neural networks is to run the MNIST dataset through a digit categorizer. Let's skip that (or you can do that on your own) and try to make another simple task. Our "hello world" project: a function approximator. We'll create a neural network that approximates any function and then use the network as a waveshaper with the `math.neural` Scriptnode node.

> The fact that I'm writing a machine learning tutorial might be the most impostory thing I've ever done considering the fact that I've spend about 7 days learning about this whole thing, so bear with me if you spot incredibly stupid things along the way

1. Download and install TorchStudio from [here](https://www.torchstudio.ai/), then install it's VS Code extension. If you've never done any ML stuff, google some tutorial and run the MNIST dataset through their hello world model to get a first impression of the basic workflow.
2. Open this folder in VS Code, then right click on the file browser and choose **New TorchStudio training**. If this entry doesn't show up, delete the VS Code `extension.json` file and restart (thank me later for saving you one hour of troubleshooting here... (If even this doesn't work out, take another route through Spyder or some other editor. You'll only need the .tsz file)). Click on the newly created `.tsz` file to open Torch Studio.
3. In the "Dataset" tab, choose "Custom Dataset", tick the "MyDataset" > **Code** box and drop the content of `dataset.py` in there, then click load. It should light up green if the loading is done. If you want, you can hack around in the code to change the function we're trying to approximate.
4. Create a Model (click the Plus sign on the tabs next to "Dataset"), choose **Custom Model**, smash that **Code** tickbox and drop `model.py` in the windows, then click **Build**. Should light up green if done.
5. Start **training** the model, but make sure to set the Hyperparameters **Batch Size** to **1**, otherwise the training will fail. I'm sure there's a smarter way of solving this (as a Batch Size of 1 feels like a very stupid thing to do, but at this current point I don't know a better way and it gets the job done). If you've got a GPU, then use it, it will accelerate the training.
6. After some training time the validation will go up to `1.0` which means that the network is fully trained and manages to approximate the function we've designed in the dataset. Now it's time to export it and load it into HISE.
7. First: Export the weights from the TorchStudio model. Click on the top-left hamburger icon, then choose **Export Model**. Choose the **Model Weights** `.pth` format (should be preselected) and overwrite the existing `_weights.pth` file.
8. Run the `exporter.py` script from the VS Code terminal with `python exporter.py` (Protip: select the Python interpreter that was installed with TorchStudio to ensure that everything is setup correctly). This should create a `model.json` file.
9. Create a neural network in HISE (`Engine.createNeuralNetwork("nn")`) and use the `nn.loadPytorchModel()` function with the JSON object you've exported.
