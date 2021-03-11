from setuptools import setup, Extension


# Bypass import numpy before running install_requires
# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class get_numpy_include:
    def __str__(self):
        import numpy
        return numpy.get_include()


module = Extension(
    'k4a_module',
    sources=['pyk4a/pyk4a.cpp'],
    include_dirs=[get_numpy_include(), '/usr/local/include/opencv4', '/usr/include', '/usr/local/include'],
    library_dirs=['/usr/local/lib', '/usr/lib/x86_64-linux-gnu'],
    libraries=['k4a', 'k4abt', 'opencv_core', 'opencv_calib3d', 'opencv_imgproc', 'turbojpeg']
)

setup(
    name='pyk4a',
    version='0.6',
    description='Python wrapper for Azure Kinect SDK',
    license='GPL-3.0',
    author='Etienne Dubeau',
    install_requires=['numpy'],
    author_email='etienne.dubeau.1@ulaval.ca',
    url='https://github.com/etiennedub/pyk4a/',
    packages=['pyk4a'],
    ext_modules=[module]
)
