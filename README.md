# WaveNet Tensorflow v2

[WaveNet](https://arxiv.org/abs/1609.03499) with TensorFlow 2.0

## Train

```bash=
   ls wavs/*.wav | tail -n+10 > train_files.txt
   ls wavs/*.wav | head -n10 > test_files.txt
   python preprocess.py
   python train.py
```

## Eval

```bash=
   python synthesize.py [input_path] [output_path] [weight_path]
```

## Weights

now training...

## References

[WaveNet](https://arxiv.org/abs/1609.03499)

論文。

[r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)  
[LESS IS MORE/WaveNet vocoder をやってみましたので、その記録です](https://r9y9.github.io/blog/2018/01/28/wavenet_vocoder/)

いくつもの論文で使われている実装。PyTorch。

[Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)

Tactron2 + WaveNetのDeepMindの人の実装。TensorFlow v1。

[Monthly Hacker's Blog/VQ-VAEの追試で得たWaveNetのノウハウをまとめてみた。](https://www.monthly-hack.com/entry/2018/02/23/203208)

WaveNetに関する知見が纏められている。

[Synthesize Human Speech with WaveNet](https://chainer-colab-notebook.readthedocs.io/ja/latest/notebook/official_example/wavenet.html)

Colabを用いた解説。Chainer。

[The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

英語の単一話者のデータセット。

[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)

日本語の単一話者のデータセット。
