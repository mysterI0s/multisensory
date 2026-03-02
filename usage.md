# Multisensory Repository: Python 3 Usage Guide

This document briefly explains how to set up and run the modernized `multisensory` codebase on Python 3.13.

## Requirements

You will need Python 3 installed. The migration specifically updated TensorFlow 1.x components to run via the compatibility wrappers available in TensorFlow 2.x. Therefore, you must install regular TensorFlow and the `tf_slim` module, which replaces `tf.contrib.slim`.

```bash
pip install tensorflow tf_slim scipy numpy pillow
```

## Running the Examples

All files should be executed natively with `python` or `python3` (depending on your OS alias). 

If you are running the pretrained shift example (as given in the original `README`), you can simply run:

```bash
python src/shift_example.py
```

This will invoke the neural network using the modernized codebase. The underlying network definitions inside `shift_net.py` and `tfutil.py` are now cleanly routed through `tensorflow.compat.v1`.

### Data Loading
If you use scripts that unpickle old Python 2 object files, remember that Python 3 defaults to ASCII encoding while unpickling. If you encounter a `UnicodeDecodeError`, alter the `pickle.load(f)` or `np.load(f)` calls in the data loaders to:
```python
pickle.load(f, encoding='latin1')
```
*(Note: Most common load routines in `aolib.util` have been kept generalized, but payload specifics may vary).*

## Core Changes Made

- **Tuple Parameter Unpacking**: All functions that previously unpacked tuples cleanly in their signatures (e.g., `def rect_area((x, y, w, h)):`) have been converted into explicit unpacking statements (e.g., `def rect_area(bbox): x, y, w, h = bbox`).
- **TF Slim**: `tensorflow.contrib.slim` handles model layers. It is completely removed in TF2, hence the requirement to pip install and use `tf_slim`.
- **Eager Execution Disabled**: Files like `tfutil.py` inherently assume a declarative graph structure (Session runs). Eager execution is disabled globally via `tf.disable_v2_behavior()`.
