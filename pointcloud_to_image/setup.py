from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['pointcloud_to_image'],
    package_dir={'pointcloud_to_image': 'src/pointcloud_to_image'}
)

setup(**d)