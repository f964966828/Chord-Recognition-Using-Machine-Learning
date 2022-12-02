# Chord-Recognition-Using-Machine-Learning

## File Usage
- params.yaml
  - parameters setting
- utils.py
  - user defined function / class
- preprocess.py
  - transfer .wav files into a json file using pitch class profiler (pcp)
  - mode = "train" / "test" (set in params.yaml)
  - `python preprocess.py`
- KNN.py, SVM.py, DecisionTree.py
  - input: train json file
  - algorithm implementation
  - `python [KNN/SVM/DecisionTree].py`
- compare.py
  - imput: train json file
  - compare performance between algorithms
  - `python compare.py`
- test.py
  - input: test json file
  - get prediction result of given file
  - `python test.py`
  
## Dataset Hierarchy
```
.
├── Dataset
│   ├── a
│   ├── am
│   ├── bm
│   ├── c
│   ├── d
│   ├── dm
│   ├── e
│   ├── em
│   ├── f
│   ├── g
│   └── data.json
├── TestWave
│   ├── about_a_girl.wav
│   ├── test.json
│   └── wave
└── params.yaml
```



