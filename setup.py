from setuptools import setup, find_packages


setup(name='reeses',
      version='0.0.3',
      author='Mark Weiss',
      author_email='mark.s.weiss.ma@gmail.com',
      description=open('README.md'),
      url='http://github.com/marksweissma/reeses',
      packages=find_packages(),
      install_requires=['sklearn', 'numpy', 'attrs'],
      summary='sklearn piecewise modeling plugin',
      license='MIT',
      zip_safe=False
      )
