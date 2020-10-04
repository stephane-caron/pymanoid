**********************
Simulation environment
**********************

Processes
=========

Simulations schedule calls to a number of "processes" into an infinite loop,
which represents the control loop of the robot. A process is a simple wrapper
around an :func:`on_tick` function, which is called at each iteration of the
simulation loop.

.. autoclass:: pymanoid.sim.Process
    :members:

Simulation
==========

The simulation object is both a global environment and a serial process
scheduler. As an environment, it is passed as argument when calling the
:func:`on_tick` functions of child processes, and also contains a number of
fields, such as ``dt`` (simulation time step) or ``gravity`` (gravity vector in
the world frame).

.. autoclass:: pymanoid.sim.Simulation
    :members:

Camera recording
================

To record a video of your simulation, schedule a camera recorder as follows:

.. code::

    sim = pymanoid.Simulation(dt=0.03)
    camera_recorder = pymanoid.CameraRecorder(sim, fname="my_video.mp4")
    sim.schedule_extra(camera_recorder)

Upon scheduling the camera recorder, the following message will appear in your
Python shell:

.. code::

   Please click on the OpenRAVE window.

The mouse pointer changes to a cross. While it is like this, click on the
OpenRAVE window (so that the recorder knows which window to record from). Then,
start or step your simulation as usual.

When your simulation is over, run the video conversion script created in the
current directory:

.. code::

    ./make_pymanoid_video.sh

After completion, the file ``my_video.mp4`` is created in the current
directory.

.. autoclass:: pymanoid.sim.CameraRecorder
    :members:

Making a new process
====================

Imagine we want to record knee angles while the robot moves in the `horizontal walking example <https://github.com/stephane-caron/pymanoid/blob/master/examples/horizontal_walking.py>`_. We can create a new process class:

.. code:: python

    class KneeAngleRecorder(pymanoid.Process):

        def __init__(self):
            """
            Initialize process. Don't forget to call parent constructor.
            """
            super(KneeAngleRecorder, self).__init__()
            self.left_knee_angles = []
            self.right_knee_angles = []
            self.times = []

        def on_tick(self, sim):
            """
            Update function run at every simulation tick.

            Parameters
            ----------
            sim : Simulation
                Instance of the current simulation.
            """
            self.left_knee_angles.append(robot.q[robot.left_knee])
            self.right_knee_angles.append(robot.q[robot.right_knee])
            self.times.append(sim.time)

To execute it, we instantiate a process of this class and add it to the
simulation right before the call to ``start_walking()``:

.. code:: python

    recorder = KneeAngleRecorder()
    sim.schedule_extra(recorder)
    start_walking()  # comment this out to start walking manually

We can plot the data recorded in this process at any time by:

.. code:: python

    import pylab
    pylab.ion()
    pylab.plot(recorder.times, recorder.left_knee_angles)
    pylab.plot(recorder.times, recorder.right_knee_angles)

Note that we scheduled the recorder as an extra process, using
``sim.schedule_extra`` rather than ``sim.schedule``, so that it does not get
counted in the computation time budget of the control loop.
