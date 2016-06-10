# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name='pymanoid',
    version='0.4.2',
    description="Python library for humanoid robotics using OpenRAVE",
    url="https://github.com/stephane-caron/pymanoid",
    author="Stéphane Caron",
    author_email="stephane.caron@normalesup.org",
    license="GPL",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 2.7'],
    packages=['pymanoid', 'pymanoid.robots', 'pymanoid.toolbox']
)
