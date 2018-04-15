from setuptools import setup, find_packages

setup(
    name='nmt',
    version='0.9.0',
    description='Python package for developing models capable of translating text from on language to another using neural network created in Keras.',
    license='MIT',
    packages=find_packages(),
    author='Jonas Holcner',
    author_email='jonas.holcner@gmail.com',
    keywords=['keras', 'tensorflow', 'nmt', 'seq2seq'],
    install_requires=['gensim', 'tensorflow_gpu', 'setuptools', 'Keras', 'numpy',
                      'nltk', 'beautifulsoup4', 'mock', 'h5py'],
    url='https://github.com/jojkos/neural-machine-translation'
)