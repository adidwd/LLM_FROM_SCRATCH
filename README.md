**#LLM_FROM_SCRATCH**

This project is inspired from Vizuara build LLM from Scratch youtube series. This project is purely for learning intentions and include quite a good amount of experimental code as well which is not used finally. This repo is still in works and the code works till training LLM on mac laptop (can be done on cloud as well and changing device type to cuda)

**Credits**

This code is inspired from teh youtube series: Build LLM from scratch by Vizuara.

Link to Series:https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu

**Module definition:**



Input_Target_Pairs: This creates the input and target data for training using opensource gpt2 enoder /decoder. create_input_output.py is the py file which is used in this project. Other files are just for learning perspective.

Multihead Attention: This is the heart of LLM. Although it is a simple python code, theory behind it is quite interesting. I would highly encourage to go through the videos to understand it.

Layernormalization: Simple and easy normalization technique used in ML training.

Feedforward: Consists of Simple Neural network with hidden layer =4x input dimension to NN and converging back to input dimension using Gelu function as its activation. Skip connection is also included in the module enabling skip connections for better NN learning.

Transformerblock: This is the place where Multihead attention, feed forward network, layer normalization and skip connections comes together. For reference, check the videos build LLM from scratch/ check attention is all you need paper for architecture diagram (how these different parts are sequenced/stitched together).

GPTModel: This consists of running the input-output-> running tranformer block-> getting output in output head of transformer and calculating loss from training and generating next set of tokens from the model. train.py is the place where GPT model (GPTmodel.py) is run and loss is calculated (calculate_loss.py) and generating next token is executed(generatenexttoken.py).

**To execute: Place your choice of txt data (I used same data as the youtube series) in root folder, change the name of txt data in train.py and Just run train.py in GPTModel folder.**

