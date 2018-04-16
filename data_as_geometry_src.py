import numpy as np
from IPython.display import display
from ipywidgets import Layout, FloatText, Box, VBox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
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
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def generate_circle_data(N=500):
    ''' Generate two classes of data '''
    N = 500
    t = np.linspace(0, 360, N)
    x1 = np.random.normal(size=N, scale=1.3)
    y1 = np.random.normal(size=N, scale=1.3)
    x2 = 6*np.cos(t) + np.random.normal(size=N, scale=0.5)
    y2 = 6*np.sin(t) + np.random.normal(size=N, scale=0.5)
    X = np.vstack((np.array([x1,y1]).T, np.array([x2, y2]).T))
    Y = np.concatenate([np.ones((N,)), np.zeros((N,))])
    return X, Y

def plot_data(X, Y):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    ax.plot(X[Y==0,0], X[Y==0,1], 'b.', label='class 1')
    ax.plot(X[Y==1,0], X[Y==1,1], 'r.', label='class 2')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def widget1():
    # True values
    # w1 = np.array([0.64, 0.41])
    # w2 = np.array([0.053, -0.71])
    # w3 = np.array([-0.67, 0.33])
    w11=FloatText(value=0.64, description=r'\(w_{11}^{(1)} \)')
    w12=FloatText(value=0.41, description=r'\(w_{12}^{(1)} \)')
    w10=FloatText(value=-1.5, description=r'\(w_{10}^{(1)} \)')
    W1 = [w11, w12, w10]

    w21=FloatText(value=0.053, description=r'\(w_{21}^{(1)} \)')
    w22=FloatText(value=-0.71, description=r'\(w_{22}^{(1)} \)')
    w20=FloatText(value=-1.5, description=r'\(w_{20}^{(1)} \)')
    W2 = [w21, w22, w20]

    w31=FloatText(value=-0.67, description=r'\(w_{31}^{(1)} \)')
    w32=FloatText(value=0.33, description=r'\(w_{32}^{(1)} \)')
    w30=FloatText(value=-1.5, description=r'\(w_{30}^{(1)} \)')
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
    n1=FloatText(value=-1.6, description=r'\(n_{1}^{(1)} \)')
    n2=FloatText(value=-1.6, description=r'\(n_{2}^{(1)} \)')
    n3=FloatText(value=-1.6, description=r'\(n_{3}^{(1)} \)')
    n4=FloatText(value=-2.2, description=r'\(offset\)')
    N = [n1, n2, n3, n4]
    box_layout = Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        width='100%')
    box = Box(children=N, layout=box_layout)
    display(box)
    return N

def plot_decision_lines(X, Y, W):
    xx=np.linspace(-8, 8, 10)
    l1_0=-(W[0,0]*xx+W[0,2])/W[0,1]
    l2_0=-(W[1,0]*xx+W[1,2])/W[1,1]
    l3_0=-(W[2,0]*xx+W[2,2])/W[2,1]
    fig = plt.figure(figsize=(5, 5)); ax=fig.gca()
    ax.plot(xx, l1_0, label='neuron 1', lw=3)
    ax.plot(xx, l2_0, label='neuron 2', lw=3)
    ax.plot(xx, l3_0, label='neuron 3', lw=3)
    ax.plot(X[Y==0,0], X[Y==0,1], 'b.', label='class 1')
    ax.plot(X[Y==1,0], X[Y==1,1], 'r.', label='class 2')
    ax.legend(bbox_to_anchor=(1, 1))
    ax.axis([-8, 8, -8, 8])
    plt.show()
    
def plot_decision_contour(X, Y, W):
    n = 100
    x, y = np.meshgrid(np.linspace(-8, 8, n), np.linspace(-8, 8, n))
    p = np.array([x.flatten(), y.flatten(), np.ones(n*n)])
    l1 = np.tanh(W[0,:].T.dot(p)).reshape(n, n)
    l2 = np.tanh(W[1,:].T.dot(p)).reshape(n, n)
    l3 = np.tanh(W[2,:].T.dot(p)).reshape(n, n)
    l = np.array([l1, l2, l3])

    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    for i in range(3):
        ax[i].plot(X[Y==0,0], X[Y==0,1], 'b.', label='class 1')
        ax[i].plot(X[Y==1,0], X[Y==1,1], 'r.', label='class 2')
        cf = ax[i].contourf(x, y, l[i], 100)
    fig.colorbar(cf, ax=ax[2])
    plt.show()
    
def plot3d_space(X, Y, W, N):
    n = 10
    X2 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    H = np.tanh(W.dot(X2.T))
    x, y = np.meshgrid(np.linspace(-1.5, 1, n), np.linspace(-1.5, 1, n))
    z = -(N[0]*x + N[1]*y + N[3]) / N[2]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_aspect('equal')
    ax.scatter(H[0,Y==0], H[1,Y==0], H[2,Y==0], c='b', label='class 1')
    ax.scatter(H[0,Y==1], H[1,Y==1], H[2,Y==1], c='r', label='class 2')
    ax.quiver(np.mean(x), np.mean(y), np.mean(z), N[0], N[1], N[2])
    ax.plot_surface(x,y,z,alpha=0.2)
    ax.set_xlabel('h1-neuron 1')
    ax.set_ylabel('h2-neuron 2')
    ax.set_zlabel('h3-neuron 3')
    set_axes_equal(ax)
    plt.legend()

    n = 100
    x, y = np.meshgrid(np.linspace(-8, 8, n), np.linspace(-8, 8, n))
    p = np.array([x.flatten(), y.flatten(), np.ones(n*n)])
    l1 = np.tanh(W[0,:].T.dot(p)).reshape(n, n)
    l2 = np.tanh(W[1,:].T.dot(p)).reshape(n, n)
    l3 = np.tanh(W[2,:].T.dot(p)).reshape(n, n)
    hid = np.array([l1.flatten(), l2.flatten(), l3.flatten()])
    hid = np.concatenate([hid, np.ones((1, hid.shape[1]))], axis=0)
    activation = np.tanh(N.T.dot(hid).reshape(n, n))
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(X[Y==0,0], X[Y==0,1], 'b.', label='class 1')
    ax.plot(X[Y==1,0], X[Y==1,1], 'r.', label='class 2')
    cf=ax.contourf(x, y, activation, 100)
    ax.legend(bbox_to_anchor=(1.6, 0.5))
    fig.colorbar(cf,ax=ax)
    plt.show()