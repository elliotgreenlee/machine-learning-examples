def read_training_data():
    file_name = "synth.tr"
    return read_data(file_name)


def read_testing_data():
    file_name = "synth.te"
    return read_data(file_name)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_data(file_name):
    X = []
    Class = []

    with open(file_name, "r") as f:
        for line in f:
            x, y, class_name = line.split()
            if isfloat(x) and isfloat(y) and isfloat(class_name):
                x_current = [float(x), float(y)]
                X.append(x_current)
                Class.append(int(class_name))

    return X, Class