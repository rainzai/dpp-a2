# Currently tested ONNX Models
Switching models requires changing these relevant parameters in both the scripts and the source code.

## [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/blob/main/model.onnx)
- VOCAB_SIZE = 30522
- BATCH = 1
- SEQ_LEN = 128
- HEAD_DIM = 768
- NUM_HEADS = 12

## [bert-large-NER](https://huggingface.co/dslim/bert-large-NER/blob/main/onnx/model.onnx)
- VOCAB_SIZE = 28996
- BATCH = 1
- SEQ_LEN = 128
- HEAD_DIM = 1024
- NUM_HEADS = 16