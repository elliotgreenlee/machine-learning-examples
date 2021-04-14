from collections import Counter

# open the reals
reals = []
priors = [0] * 4
with open('realresults.txt') as f:
    for line in f:
        words = line.split()
        num = int(words[0])
        reals.append(num)
        priors[num] += 1

for index, prior in enumerate(priors):
    priors[index] /= len(reals)

# individually open each file and store in a list of predicteds
predicted_bpnn = []
with open('BPNN_results.txt') as f:
    for line in f:
        words = line.split()
        num = int(words[0])
        predicted_bpnn.append(num)

predicted_decisiontree = []
with open('decisiontree_results.txt') as f:
    for line in f:
        words = line.split()
        num = int(words[0])
        predicted_decisiontree.append(num)

predicted_mpp = []
with open('mpp_results.txt') as f:
    for line in f:
        words = line.split()
        num = int(words[0])
        predicted_mpp.append(num)

predicted_svm = []
with open('svm_results.txt') as f:
    for line in f:
        words = line.split()
        num = int(words[0])
        predicted_svm.append(num)


# Voting
# run through zip of all values
'''
final = []
for p1, p2, p3, p4, in zip(predicted_bpnn, predicted_decisiontree, predicted_mpp, predicted_svm):
    # at each sample, find most picked one
    data = Counter([p1, p2, p3, p4])
    bests = data.most_common()

    # if there is a tie, use priors
    if len(bests) > 1 and bests[0][1] == bests[1][1]:
        if priors[bests[0][0]] >= priors[bests[1][0]]:
            most = bests[0][0]
        else:
            most = bests[1][0]
    else:
        most = bests[0][0]

    # get a new prediction
    final.append(most)

with open('votingfusion_results.txt', 'w') as f:
    for sample in final:
        f.write('{}\n'.format(sample))
'''

# BKS

# get combinations of all possible predicted
combinations = []
for i in range(0, 4):
    for j in range(0, 4):
        for k in range(0, 4):
            for l in range(0, 4):
                combination = [i, j, k, l]
                combinations.append(combination)

class_counter = [[0, 0, 0, 0] for x in range(len(combinations))]
# run through zip of all values
for p1, p2, p3, p4, real in zip(predicted_bpnn, predicted_decisiontree, predicted_mpp, predicted_svm, reals):
    # at each sample, find which combination it belongs to
    index = combinations.index([p1, p2, p3, p4])
    # add to appropriate real class counter for that combination
    class_counter[index][real] += 1

# for each combination, pick the highest class from the real class counter,
# and create a lookup table (maybe another tuple?)
predictions = [0] * len(combinations)
for index, count in enumerate(class_counter):
    predictions[index] = count.index(max(count))

# run through zip of all values
final = []
for p1, p2, p3, p4 in zip(predicted_bpnn, predicted_decisiontree, predicted_mpp, predicted_svm):
    # at each sample, find which combination it belongs to
    final.append(predictions[combinations.index([p1, p2, p3, p4])])

with open('bksfusion_results.txt', 'w') as f:
    for sample in final:
        f.write('{}\n'.format(sample))


# compare to real
correct = 0
for real, predicted in zip(reals, final):
    if real == predicted:
        correct += 1

# total accuracy
accuracy = (1.0 * correct) / len(reals)
print(accuracy)


