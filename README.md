# warpctc module for PyTorch

*Thomas Viehmann <tv@lernapparat.de>*

This is a warpctc module for PyTorch.
It targets PyTorch 0.4+ and uses the C++ extension mechanism.

Warp-CTC was created by Baidu, see [the original README](README.orig.md)

The idea is to

```
cd pytorch_bindings
python3 setup.py bdist_wheel
```
and install the wheel in `pytorch_bindings/dist`. Remember to set the
compiler as you do to compile PyTorch.
You can test with
```
python3 tests/test.py -v
```
