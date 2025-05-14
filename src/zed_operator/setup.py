from setuptools import find_packages, setup
import os
import glob

package_name = 'zed_operator'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ade157',
    maintainer_email='ade157@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed_operator = zed_operator.zed_operator:main',
        ],
    },
)
