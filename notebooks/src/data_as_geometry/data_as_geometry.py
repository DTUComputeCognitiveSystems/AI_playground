import warnings

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Layout, FloatText, Box, VBox, FloatSlider, HBox


class ProjectionPlotter:
    def __init__(self, starting_values=None, limits=(-3.0, 5.0), linewidth=3):
        self.linewidth = linewidth
        self.starting_values = starting_values
        self.limits = limits

        # Default values
        if starting_values is None:
            starting_values = [
                [1, 2],
                [3, 1]
            ]

        # Make widgets
        self.widgets = [
            FloatSlider(
                value=val,
                min=limits[0],
                max=limits[1],
                step=0.1,
                description='${}_{{{}}}$'.format(["v", "w"][nr], d),
                disabled=False,
                continuous_update=True,
                orientation='vertical',
                readout=True,
                readout_format='.1f',
            )
            for nr, vector in enumerate(starting_values)
            for d, val in enumerate(vector)
        ]

        # Make widget box
        self.box = HBox(self.widgets)

        # Set fields
        self.fig = self.ax = None

        # Make figure when widgets are shown
        self.box.on_displayed(self.plot_function)

        # Make widgets pass data to plotting function
        for widget in self.widgets:
            widget.observe(self.plot_function)

    def plot_function(self, _=None):
        v1 = np.array([self.widgets[0].value, self.widgets[1].value])
        v2 = np.array([self.widgets[2].value, self.widgets[3].value])

        # Try to do projection
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            projection = v1.dot(v2) * v2 / v2.dot(v2)

            # Check if we succeeded
            if len(warn) > 0 and issubclass(warn[-1].category, RuntimeWarning):
                projection = np.array([0, 0])

        # Initialize figure
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = plt.gca()
        else:
            self.ax.cla()

        # Range of plot
        plot_range = self.limits[1] - self.limits[0]

        # Arrow settings
        arrow_settings = [
            (v1, "$v$", "b", self.linewidth),
            (v2, "$w$", "k", self.linewidth),
            (projection, "$p$", "g", self.linewidth),
        ]

        # Make arrows
        arrows = []
        names = []
        for vector, name, color, linewidth in arrow_settings:
            if sum([abs(val) for val in vector]) >= 0.1:
                arrows.append(self.ax.arrow(
                    0, 0, vector[0], vector[1],
                    head_width=0.02 * plot_range,
                    head_length=0.03 * plot_range,
                    fc=color,
                    ec=color,
                    linewidth=linewidth,
                ))
                arrows[-1].set_capstyle('round')
                names.append(name)

        # Set axes
        self.ax.set_xlim(*self.limits)
        self.ax.set_ylim(*self.limits)
        self.ax.set_aspect("equal")

        # Title and legend
        self.ax.set_title("Projections")
        self.ax.legend(
            arrows,
            names,
            loc="lower left"
        )

    def start(self):
        return self.box


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def widget0():
    # True value
    # n = [-2, 1]
    alpha = FloatText(value=-2, description=r'\(\alpha \)')
    beta = FloatText(value=1, description=r'\(\beta\)')
    w = [alpha, beta]
    box_layout = Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        width='100%')
    box = Box(children=w, layout=box_layout)
    display(box)
    return w


class exercise_2_4:
    def __init__(self, Widget):
        self.widget = Widget
        for val in self.widget:
            val.observe(self.display)
        self.fig = plt.figure(figsize=(10, 5))
        self.ax = plt.gca()

        # Generate data
        self.x1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], size=500)
        self.x2 = np.random.multivariate_normal([-2, 0], [[1, 0], [0, 1]], size=500)
        self.xmin = np.minimum(self.x1[:, 0].min(), self.x2[:, 0].min())
        self.xmax = np.maximum(self.x1[:, 0].max(), self.x2[:, 0].max())
        self.ymin = np.minimum(self.x1[:, 1].min(), self.x2[:, 1].min())
        self.ymax = np.maximum(self.x1[:, 1].max(), self.x2[:, 1].max())

        self.display()

    def display(self, _=None):
        self.ax.cla()
        alpha, beta = [w.value for w in self.widget]
        x = np.linspace(self.xmin, self.xmax, 100)
        y = alpha * x + beta
        idx = np.logical_and(self.ymax > y, y > self.ymin)
        x = x[idx]
        y = y[idx]
        # Plot data
        self.ax.plot(self.x1[:, 0], self.x1[:, 1], 'b.')
        self.ax.plot(self.x2[:, 0], self.x2[:, 1], 'r.')
        self.ax.plot(x, y, '-k')
        self.ax.axis('equal')
        self.ax.grid()
        normalize = (alpha ** 2 + beta ** 2)
        self.ax.arrow(np.mean(x), np.mean(y), -alpha / normalize, beta / normalize,
                      head_width=0.5, head_length=0.5, fc='k', ec='k')
        plt.draw()


def generate_circle_data(N=500):
    """ Generate two classes of data """
    N = 500
    t = np.linspace(0, 360, N)
    x1 = np.random.normal(size=N, scale=1.3)
    y1 = np.random.normal(size=N, scale=1.3)
    x2 = 6 * np.cos(t) + np.random.normal(size=N, scale=0.5)
    y2 = 6 * np.sin(t) + np.random.normal(size=N, scale=0.5)
    X = np.vstack((np.array([x1, y1]).T, np.array([x2, y2]).T))
    Y = np.concatenate([np.ones((N,)), np.zeros((N,))])
    return X, Y


def plot_data(X, Y):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.plot(X[Y == 0, 0], X[Y == 0, 1], 'b.', label='class 1')
    ax.plot(X[Y == 1, 0], X[Y == 1, 1], 'r.', label='class 2')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    plt.show()


def widget1():
    # True values
    # w1 = np.array([0.64, 0.41])
    # w2 = np.array([0.053, -0.71])
    # w3 = np.array([-0.67, 0.33])
    w11 = FloatText(value=0.64, description=r'\(w_{11}^{(1)} \)')
    w12 = FloatText(value=0.41, description=r'\(w_{12}^{(1)} \)')
    w10 = FloatText(value=-1.5, description=r'\(w_{10}^{(1)} \)')
    W1 = [w11, w12, w10]

    w21 = FloatText(value=0.053, description=r'\(w_{21}^{(1)} \)')
    w22 = FloatText(value=-0.71, description=r'\(w_{22}^{(1)} \)')
    w20 = FloatText(value=-1.5, description=r'\(w_{20}^{(1)} \)')
    W2 = [w21, w22, w20]

    w31 = FloatText(value=-0.67, description=r'\(w_{31}^{(1)} \)')
    w32 = FloatText(value=0.33, description=r'\(w_{32}^{(1)} \)')
    w30 = FloatText(value=-1.5, description=r'\(w_{30}^{(1)} \)')
    W3 = [w31, w32, w30]
    box_layout = Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        width='100%')
    box_1 = Box(children=W1, layout=box_layout)
    box_2 = Box(children=W2, layout=box_layout)
    box_3 = Box(children=W3, layout=box_layout)
    display(VBox([box_1, box_2, box_3]))
    return [W1, W2, W3]


def widget2():
    # True value
    # n = [-1.6, -1.6, -1.6, -2.2]
    n1 = FloatText(value=-1.6, description=r'\(n_{1}^{(1)} \)')
    n2 = FloatText(value=-1.6, description=r'\(n_{2}^{(1)} \)')
    n3 = FloatText(value=-1.6, description=r'\(n_{3}^{(1)} \)')
    n4 = FloatText(value=-2.2, description=r'\(offset\)')
    N = [n1, n2, n3, n4]
    box_layout = Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        width='100%')
    box = Box(children=N, layout=box_layout)
    display(box)
    return N


class plot_decision_lines:
    def __init__(self, X, Y, Widget):
        self.widget = Widget
        self.X = X
        self.Y = Y

        # Check for updates
        for val in self.widget:
            for w in val:
                w.observe(self.display)

        # Initilize figure
        self.fig = plt.figure(figsize=(10, 5));
        self.ax = self.fig.gca()

        # Display initlial settings
        self.display()

    def display(self, _=None):
        self.ax.cla()
        # Extract values
        W = np.array([[w.value for w in self.widget[0]],
                      [w.value for w in self.widget[1]],
                      [w.value for w in self.widget[2]]])
        W = (W.T / np.linalg.norm(W, axis=1)).T
        # Generate lines
        xx = np.linspace(-8, 8, 20)
        l1_0 = -(W[0, 0] * xx + W[0, 2]) / W[0, 1]
        l2_0 = -(W[1, 0] * xx + W[1, 2]) / W[1, 1]
        l3_0 = -(W[2, 0] * xx + W[2, 2]) / W[2, 1]

        # Plot lines
        self.ax.plot(self.X[self.Y == 0, 0], self.X[self.Y == 0, 1], 'b.', label='class 1')
        self.ax.plot(self.X[self.Y == 1, 0], self.X[self.Y == 1, 1], 'r.', label='class 2')
        self.ax.plot(xx, l1_0, label='neuron 1', lw=3)
        self.ax.plot(xx, l2_0, label='neuron 2', lw=3)
        self.ax.plot(xx, l3_0, label='neuron 3', lw=3)

        # Plot normal vectors
        inside = np.logical_and(l1_0 > -8, l1_0 < 8)
        self.ax.arrow(np.mean(xx[inside]), np.mean(l1_0[inside]), 2 * W[0, 0], 2 * W[0, 1],
                      head_width=0.5, head_length=0.5, fc='k', ec='k')
        inside = np.logical_and(l2_0 > -8, l2_0 < 8)
        self.ax.arrow(np.mean(xx[inside]), np.mean(l2_0[inside]), 2 * W[1, 0], 2 * W[1, 1],
                      head_width=0.5, head_length=0.5, fc='k', ec='k')
        inside = np.logical_and(l3_0 > -8, l3_0 < 8)
        self.ax.arrow(np.mean(xx[inside]), np.mean(l3_0[inside]), 2 * W[2, 0], 2 * W[2, 1],
                      head_width=0.5, head_length=0.5, fc='k', ec='k')

        self.ax.legend(bbox_to_anchor=(1, 1))
        self.ax.axis('equal')
        self.ax.axis([-8, 8, -8, 8])

        plt.draw()


class plot_decision_contour:
    def __init__(self, X, Y, Widget):
        self.X = X
        self.Y = Y
        self.widget = Widget

        for val in self.widget:
            for w in val:
                w.observe(self.display)

        self.n = 10
        self.x, self.y = np.meshgrid(np.linspace(-8, 8, self.n), np.linspace(-8, 8, self.n))
        self.p = np.array([self.x.flatten(), self.y.flatten(), np.ones(self.n * self.n)])

        # Set figure
        self.fig, self.ax = plt.subplots(1, 3, figsize=(10, 5))

        # Display initial settings
        self.display()

    def display(self, _=None):
        for i in range(3):
            self.ax[i].cla()  # clear figure

        # Get weights
        W = np.array([[w.value for w in self.widget[0]],
                      [w.value for w in self.widget[1]],
                      [w.value for w in self.widget[2]]])

        # Calculate activations
        l1 = np.tanh(W[0, :].T.dot(self.p)).reshape(self.n, self.n)
        l2 = np.tanh(W[1, :].T.dot(self.p)).reshape(self.n, self.n)
        l3 = np.tanh(W[2, :].T.dot(self.p)).reshape(self.n, self.n)
        l = np.array([l1, l2, l3])

        for i in range(3):
            self.ax[i].set_title('Neuron ' + str(i + 1) + ' activation', fontsize=15)
            self.ax[i].plot(self.X[self.Y == 0, 0], self.X[self.Y == 0, 1], 'b.', label='class 1')
            self.ax[i].plot(self.X[self.Y == 1, 0], self.X[self.Y == 1, 1], 'r.', label='class 2')
            cf = self.ax[i].contourf(self.x, self.y, l[i], 100)
        plt.show()


class plot3d_space:
    def __init__(self, X, Y, Widget1, Widget2):
        self.widget1 = Widget1  # weights
        self.widget2 = Widget2  # normal vector
        self.X = X
        self.Y = Y

        # Check for updates
        for val in self.widget2:
            val.observe(self.display)

        # Initilize figure
        self.fig = plt.figure(figsize=(10, 10))
        self.ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, projection='3d')
        self.ax2 = plt.subplot2grid((2, 2), (1, 0))
        self.ax3 = plt.subplot2grid((2, 2), (1, 1))

        # Display initlial settings
        self.display()

    def display(self, _=None):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()

        # Get weights
        W = np.array([[w.value for w in self.widget1[0]],
                      [w.value for w in self.widget1[1]],
                      [w.value for w in self.widget1[2]]])
        N = np.array([n.value for n in self.widget2])

        n = 10
        X2 = np.concatenate([self.X, np.ones((self.X.shape[0], 1))], axis=1)
        H = np.tanh(W.dot(X2.T))
        x, y = np.meshgrid(np.linspace(-1.5, 1, n), np.linspace(-1.5, 1, n))
        z = -(N[0] * x + N[1] * y + N[3]) / N[2]

        self.point_act = N.dot(np.concatenate([H, np.ones((1, H.shape[1]))], axis=0))

        self.ax1.set_aspect('equal')
        self.ax1.scatter(H[0, self.Y == 0], H[1, self.Y == 0], H[2, self.Y == 0], c='b', label='class 1')
        self.ax1.scatter(H[0, self.Y == 1], H[1, self.Y == 1], H[2, self.Y == 1], c='r', label='class 2')
        self.ax1.quiver(np.mean(x), np.mean(y), np.mean(z), N[0], N[1], N[2])
        self.ax1.plot_surface(x, y, z, alpha=0.2)
        self.ax1.set_xlabel('h1-neuron 1')
        self.ax1.set_ylabel('h2-neuron 2')
        self.ax1.set_zlabel('h3-neuron 3')
        set_axes_equal(self.ax1)
        plt.legend()

        n = 50
        x, y = np.meshgrid(np.linspace(-8, 8, n), np.linspace(-8, 8, n))
        p = np.array([x.flatten(), y.flatten(), np.ones(n * n)])
        l1 = np.tanh(W[0, :].T.dot(p)).reshape(n, n)
        l2 = np.tanh(W[1, :].T.dot(p)).reshape(n, n)
        l3 = np.tanh(W[2, :].T.dot(p)).reshape(n, n)
        hid = np.array([l1.flatten(), l2.flatten(), l3.flatten()])
        hid = np.concatenate([hid, np.ones((1, hid.shape[1]))], axis=0)
        activation = np.tanh(N.T.dot(hid).reshape(n, n))

        self.ax2.plot(self.X[self.Y == 0, 0], self.X[self.Y == 0, 1], 'b.', label='class 1')
        self.ax2.plot(self.X[self.Y == 1, 0], self.X[self.Y == 1, 1], 'r.', label='class 2')
        cf = self.ax2.contourf(x, y, activation, 100)
        self.ax2.legend(bbox_to_anchor=(1.6, 0.5))

        red_class = activation < 0
        self.ax3.plot(self.X[self.Y == 0, 0], self.X[self.Y == 0, 1], 'b.', label='class 1')
        self.ax3.plot(self.X[self.Y == 1, 0], self.X[self.Y == 1, 1], 'r.', label='class 2')
        self.ax3.contourf(x, y, red_class, 100, cmap='RdBu')
        plt.show()

    def get_res(self):
        point_cla = self.point_act > 0
        Yb = np.bool8(self.Y)
        print('Results:')
        print('-------------------------------------------------')
        print('Number of red points classified as red:  ', np.sum(point_cla[Yb] == Yb[Yb]))
        print('Number of red points classified as blue: ', np.sum(point_cla[Yb] != Yb[Yb]))
        print('Number of blue points classified as blue:', np.sum(point_cla[~Yb] == Yb[~Yb]))
        print('Number of blue points classifies as red: ', np.sum(point_cla[~Yb] != Yb[~Yb]))
        print('-------------------------------------------------')
