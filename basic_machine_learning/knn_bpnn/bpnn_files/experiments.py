"""
Elliot Greenlee

571 Project 3

April 6, 2017

knn and backpropagation neural network
"""

from bpnn import *

# XOR Samples
possible_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
possible_outputs = [0.0, 1.0, 1.0, 0.0]

'''
print '========================='
print 'Different Initializations'
print '========================='
'''

'''
print 'Weights Initialized to 0'

bpnn = BPNN()
bpnn.initialize_weights_0()

for epoch in range(1, 1000000 + 1):
    if epoch % 100000 == 0:
        print epoch

    for i in range(0, 4):
        choice = random.randint(0, 3)

        output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.sigmoid)
        bpnn.backwards_propagation(possible_outputs[choice])

    done = True
    for i in range(0, 4):
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
        error = bpnn.total_error(output, possible_outputs[i])
        if error >= 0.005:
            done = False

    if done:
        break

print 'Final'
print epoch
print bpnn.weights
for i in range(0, 4):
    print i, ':', possible_inputs[i][0], possible_inputs[i][1]
    output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
    print output
    print 'Error:', bpnn.total_error(output, possible_outputs[i])
'''

'''
print 'Weights Initialized to a Constant in a Range'

for constant_weight in range(10, 11, 1):
    print 'Weight: ', constant_weight/10.0

    bpnn = BPNN()
    bpnn.initialize_weights_constant(constant_weight/10.0)

    for epoch in range(1, 700000 + 1):

        if epoch % 100000 == 0:
            print epoch

        for i in range(0, 4):
            choice = random.randint(0, 3)

            output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.sigmoid)
            bpnn.backwards_propagation(possible_outputs[choice])

        done = True
        for i in range(0, 4):
            output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
            error = bpnn.total_error(output, possible_outputs[i])
            if error >= 0.005:
                done = False

        if done:
            break

    print 'Final'
    print epoch
    print bpnn.weights
    for i in range(0, 4):
        print i, ':', possible_inputs[i][0], possible_inputs[i][1]
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
        print output
        print 'Error:', bpnn.total_error(output, possible_outputs[i])
'''

'''
print 'Weights Initialized Uniformly Distributed'

bpnn = BPNN()
bpnn.initialize_weights_distributed()

for epoch in range(1, 1000000 + 1):

    if epoch % 100000 == 0:
        print epoch

    for i in range(0, 4):
        choice = random.randint(0, 3)

        output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.sigmoid)
        bpnn.backwards_propagation(possible_outputs[choice])

    done = True
    for i in range(0, 4):
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
        error = bpnn.total_error(output, possible_outputs[i])
        if error >= 0.005:
            done = False

    if done:
        break

print 'Final'
print epoch
print bpnn.weights
for i in range(0, 4):
    print i, ':', possible_inputs[i][0], possible_inputs[i][1]
    output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
    print output
    print 'Error:', bpnn.total_error(output, possible_outputs[i])
'''

'''
print '========================'
print 'Different Learning Rates'
print '========================'

#10, 3, 1, 0.3, 0.1,
learning_rates = [0.03, 0.01, 0.003]
for learning_rate in learning_rates:
    print 'Learning Rate: ', learning_rate
    bpnn = BPNN()
    bpnn.learning_rate = learning_rate
    bpnn.initialize_weights_distributed()

    for epoch in range(1, 1000000 + 1):
        if epoch % 100000 == 0:
            print epoch

        for i in range(0, 4):
            choice = random.randint(0, 3)

            output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.sigmoid)
            bpnn.backwards_propagation(possible_outputs[choice])

        done = True
        for i in range(0, 4):
            output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
            error = bpnn.total_error(output, possible_outputs[i])
            if error >= 0.005:
                done = False

        if done:
            break

    print 'Final'
    print epoch
    print bpnn.weights
    for i in range(0, 4):
        print i, ':', possible_inputs[i][0], possible_inputs[i][1]
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
        print output
        print 'Error:', bpnn.total_error(output, possible_outputs[i])
'''

print '=============================='
print 'Different Activation Functions'
print '=============================='

'''
print 'Signum'
bpnn = BPNN()
bpnn.initialize_weights_distributed()

for epoch in range(1, 100000 + 1):

    if epoch % 100000 == 0:
        print epoch

    for i in range(0, 4):
        choice = random.randint(0, 3)

        output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.signum)
        bpnn.backwards_propagation(possible_outputs[choice], bpnn.d_signum)

    done = True
    for i in range(0, 4):
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.signum)
        error = bpnn.total_error(output, possible_outputs[i])
        if error >= 0.005:
            done = False

    if done:
        break

print 'Final'
print epoch
print bpnn.weights
for i in range(0, 4):
    print i, ':', possible_inputs[i][0], possible_inputs[i][1]
    output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
    print output
    print 'Error:', bpnn.total_error(output, possible_outputs[i])
'''

'''
print 'Sigmoid'
bpnn = BPNN()
bpnn.initialize_weights_distributed()

for epoch in range(1, 300000 + 1):

    if epoch % 100000 == 0:
        print epoch

    for i in range(0, 4):
        choice = random.randint(0, 3)

        output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.sigmoid)
        bpnn.backwards_propagation(possible_outputs[choice], bpnn.d_sigmoid)

    done = True
    for i in range(0, 4):
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
        error = bpnn.total_error(output, possible_outputs[i])
        if error >= 0.005:
            done = False

    if done:
        break

print 'Final'
print epoch
print bpnn.weights
for i in range(0, 4):
    print i, ':', possible_inputs[i][0], possible_inputs[i][1]
    output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.sigmoid)
    print output
    print 'Error:', bpnn.total_error(output, possible_outputs[i])
'''

'''
print 'Improved Sigmoid'
bpnn = BPNN()
bpnn.initialize_weights_distributed()

for epoch in range(1, 100000 + 1):

    if epoch % 100000 == 0:
        print epoch

    for i in range(0, 4):
        choice = random.randint(0, 3)

        output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.improved_sigmoid)
        # print choice, output
        bpnn.backwards_propagation(possible_outputs[choice], bpnn.d_improved_sigmoid)

    done = True
    for i in range(0, 4):
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.improved_sigmoid)
        error = bpnn.total_error(output, possible_outputs[i])
        if error >= 0.005:
            done = False

    if done:
        break

print 'Final'
print epoch
print bpnn.weights
for i in range(0, 4):
    print i, ':', possible_inputs[i][0], possible_inputs[i][1]
    output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.improved_sigmoid)
    print output
    print 'Error:', bpnn.total_error(output, possible_outputs[i])

'''

print '==============='
print 'Use of Momentum'
print '==============='
'''
print 'No Momentum'
bpnn = BPNN()
bpnn.initialize_weights_distributed()

for epoch in range(1, 100000 + 1):

    if epoch % 100000 == 0:
        print epoch

    choice = random.randint(0, 3)

    output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.improved_sigmoid)
    bpnn.backwards_propagation(possible_outputs[choice], bpnn.d_improved_sigmoid, bpnn.update)

    done = True
    for i in range(0, 4):
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.improved_sigmoid)
        error = bpnn.total_error(output, possible_outputs[i])
        if error >= 0.005:
            done = False

    if done:
        break

print 'Final'
print epoch
print bpnn.weights
for i in range(0, 4):
    print i, ':', possible_inputs[i][0], possible_inputs[i][1]
    output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.improved_sigmoid)
    print output
    print 'Error:', bpnn.total_error(output, possible_outputs[i])
'''

print 'Momentum'
bpnn = BPNN()
bpnn.initialize_weights_distributed()

for epoch in range(1, 1000000 + 1):

    if epoch % 100000 == 0:
        print epoch

    for i in range(0, 4):
        choice = random.randint(0, 3)

        output = bpnn.forward_propagation(possible_inputs[choice][0], possible_inputs[choice][1], bpnn.improved_sigmoid)
        bpnn.backwards_propagation(possible_outputs[choice], bpnn.d_improved_sigmoid, bpnn.update)

    done = True
    for i in range(0, 4):
        output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.improved_sigmoid)
        error = bpnn.total_error(output, possible_outputs[i])
        if error >= 0.005:
            done = False

    if done:
        break

print 'Final'
print epoch
print bpnn.weights
for i in range(0, 4):
    print i, ':', possible_inputs[i][0], possible_inputs[i][1]
    output = bpnn.forward_propagation(possible_inputs[i][0], possible_inputs[i][1], bpnn.improved_sigmoid)
    print output
    print 'Error:', bpnn.total_error(output, possible_outputs[i])





