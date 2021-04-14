import os
'''
for i in range(10, 11):
    k = i * 2
    knn = []
    with open('knnresults_normalized_{}_2.txt'.format(k)) as f:
        for line in f:
            words = line.split()
            type = int(words[0])
            if type == 4:
                type = 0
            knn.append(type)

    with open('knnresults{}_2.txt'.format(k), 'w') as f:
        for sample in knn:
            f.write('{}\n'.format(sample))
'''

reals = []
with open('realresults.txt') as f:
    for line in f:
        words = line.split()
        reals.append(int(words[0]))

files = os.listdir('.')
for file in files:
    confusion = [[0 for x in range(4)] for y in range(4)]
    predicteds = []
    with open(file) as f:
        for line in f:
            words = line.split()
            predicteds.append(int(words[0]))

    correct = 0
    for real, predicted in zip(reals, predicteds):
        confusion[real][predicted] += 1
        if real == predicted:
            correct += 1

    accuracy = (1.0 * correct) / len(predicteds)

    print(file, accuracy, confusion)

