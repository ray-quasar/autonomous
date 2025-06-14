from setuptools import setup

package_name = 'disparityExtender'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sanjot Singh',
    maintainer_email='your_email@example.com',
    description='Disparity Extender algorithm implementation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'disparityExtender = disparityExtender.disparityExtender:main',
        ],
    },
)
