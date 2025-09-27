inputs = [1, 2, 3, 2.5]                 #outputs for 3 previous layers
weights = [[0.2, 0.8, -0.5, 1.0],       #each output has a weight
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]  
bias = [2, 3, 0.5]                      #bias of current neuron




output = []
for b, weight, in zip(bias, weights):
    temp = b
    for i, w in zip(inputs, weight):
        temp += i * w
    output.append(temp)

print(output)
