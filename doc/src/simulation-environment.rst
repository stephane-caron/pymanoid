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
