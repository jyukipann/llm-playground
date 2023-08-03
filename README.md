# llm
 llm playground

# installation
wslもしくはLinuxを想定。

仮想環境
```
conda create -n hf python=3.10
conda activate hf
```
必要なライブラリ
```
pip install torch torchvision torchaudio 
pip install transformers[sentencepiece]
pip install bitsandbytes accelerate
pip install scipy
pip install flask
```
# 動かす
```bash
python chatbot_server.py
```
