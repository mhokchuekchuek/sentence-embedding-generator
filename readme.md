# Sentence Embedding generator
This python library use [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as sentence embedding generator

### variable and return type
**variable**\
`model_path [str]:` path that collect embedding generator model.\
`tokenizer_path [str]:` path that collect embedding generator tokenizer.

**return type**\
`get_embedding_from_sentence Union[str, List[str]]:` np.ndarray

### How to use
```python
from embedding_generator.usecase.engine import GeneratorEngine
# init wave tensor
example_text = "hello world"

generator = GeneratorEngine(model_path="PATH TO MODEL", tokenizer_path="PATH TO TOKENIZER")
generator.get_embedding_from_sentence(example_text)

#return value
array([[-3.44772749e-02,  3.10231652e-02,  6.73503987e-03,...,]], dtype=float32)
```
