# NMT package
Python package for neural machine translation built with Keras and Tensorflow (current version xxx?)

**nmt** is library that was developed as part of [diploma thesis](https://github.com/jojkos/master-thesis). It can train LSTM encoder-decoder model based on provided datasets. Even though machine translation is its primary use, it can be used for any other similar training, such as to train chat bots. It only depends on the provided datasets from which the model learns how to response to given sequence. 

## Installation
One of those methods can be used:
- Install package globally from wheel in dist folder (`pip install dist/nmt.whl`)
- Install package globally with python `setup.py install`
- Put it in the project folder

Use it with `import nmt`


## Parameters and usage
Complete documentation of Translator api is [here](https://rawgit.com/jojkos/neural-machine-translation/master/docs/_build/index.html#module-nmt.translator)

Exmaple of script running nmt library [main.py](https://github.com/jojkos/master-thesis/blob/master/code/main.py)
```
TODO show how tun run the main.py
```

## Documentation
- Prebuilt documentation is in docs/_build/
- Can be build from source with `sphinx-build -b html ./docs ./docs/_build`
- [Live version](https://rawgit.com/jojkos/neural-machine-translation/master/docs/_build/index.html) from github 

## Tests
- Test are written using pytest
- Can be run with `pytest tests` 

## TODOs
- [ ] better decomposition
- [ ] upload to PyPi