from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['subt_world_generation', 'subt_dataset_generation'],
    package_dir={'': 'src'}
)

setup(**d)