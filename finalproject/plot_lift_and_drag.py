import matplotlib.pyplot as plt

fname = "max_lift_drag.txt"
with open(fname) as f:
    data = f.readlines()
data = [x.strip() for x in data]
split_line = data[0].split()
x_axis_label = split_line[0]
y_axis1_label = split_line[1]
y_axis2_label = split_line[2]

thetas = []
drags = []
lifts = []
for i in range(1,len(data)):
    split_data = data[i].split()
    thetas.append(float(split_data[0]))
    drags.append(float(split_data[1]))
    lifts.append(float(split_data[2]))

def plot_max_functionals(thetas, y_data, type):
    plt.plot(thetas,y_data)
    plt.xlabel(x_axis_label)
    plt.ylabel(type)
    plt.title("Max " + type + "vs. theta")
    plt.savefig("max" + type + ".png")
    plt.clf()

def plot_both(thetas, y_data1, type1, y_data2, type2):
    plt.plot(thetas,y_data1,label=type1)
    plt.semilogy(thetas,y_data2, label=type2)
    plt.xlabel(x_axis_label)
    plt.ylabel("Lift and Drag")
    plt.title("Max lift and drag vs. theta")
    plt.legend(loc='upper left')
    plt.savefig("max_both.png")
    plt.clf()

plot_max_functionals(thetas, drags, y_axis1_label)
plot_max_functionals(thetas, lifts, y_axis2_label)
plot_both(thetas, drags,y_axis1_label, lifts, y_axis2_label)
