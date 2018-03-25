# warpctc module for PyTorch

*Thomas Viehmann <tv@lernapparat.de>*

This is a warpctc module for PyTorch.
It targets PyTorch 0.4+ and uses the C++ extension mechanism.

Warp-CTC was created by Baidu, see [the original
README](README.orig.md) and
[github project](https://github.com/baidu-research/warp-ctc).

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

Note that this is not the same as [Sean Naren's Warp-CTC PyTorch
wrapper](https://github.com/SeanNaren/warp-ctc).
I have tried to make the bindings look as PyTorch-y as possible in terms
of interface and also to stay close to the modern way of doing things
in the internals and use ATen - this had not been available when Sean
made bindings.
