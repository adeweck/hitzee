from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='hitzee',
      description='A simple package to normalise plate-based assays with suspected position biases',
      long_description=long_description,
      long_description_content_type="text/markdown",
      version="0.1",
      url='https://github.com/adeweck/hitzee',
      author='Antoine de Weck',
      author_email='adeweck@gmx.net',
      license='MIT',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      packages=['hitzee'],
      install_requires=['setuptools','numpy', 'pandas', 'scipy', 'seaborn'],
      python_requires='>=3.4',
      entry_points={
          'console_scripts': [
              'hitZ = hitzee.hitzee:hitzee'
          ]
      },
      zip_safe=False)