# bachelor-causal-extraction

To set up environment:
```
pip install -U spacy-nightly --pre
pip install -U spacy[cuda116]
python -m pip install spacy-nightly[thinc]
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python -m spacy download en_core_web_trf
python -m pip install -U spacy transformers
python -m spacy project run all_gpu
```
(not sure about order of commands)
