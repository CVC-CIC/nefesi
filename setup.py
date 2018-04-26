from setuptools import setup, find_packages

setup(
    name='nefesi',
    version='1.0.4',
    packages=find_packages(),
    url='https://github.com/ramonbal/NEFESI',
    license='GNU',
    author='oprades',
    author_email='oskarpras@gmail.com',
    description='CNN analysis',
    install_requires=[
        'tensorflow >= 1.1.0',
        'keras >= 2.0.6',
        'numpy >= 1.13.1',
        'scipy >= 0.19.1'],
)
