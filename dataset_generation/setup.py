from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['world_file_generation'],
    package_dir={'': 'src'}
)

setup(**d)