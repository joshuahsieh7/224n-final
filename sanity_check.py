import torch

from models.gpt2 import GPT2Model

from transformers import GPT2Model as OpenAIGPT2Model
from utils import model_size_to_params

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


def test_gpt2(model_size='gpt2'):
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

  # Load both the OpenAI and your own model.
  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))

  # PEFT
  peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1
  )
  model = get_peft_model(openai_model, peft_config)
  model.print_trainable_parameters()
  
  outputs = gpt(sent_ids, att_mask)
  openai_outputs = openai_model(input_ids=sent_ids, attention_mask=att_mask, output_hidden_states=True).hidden_states[-1]

  att_mask = att_mask.unsqueeze(-1)
  outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
  openai_outputs *= att_mask

  assert torch.allclose(outputs['last_hidden_state'], openai_outputs, atol=1e-1, rtol=1e-2)

  print("Your GPT2 implementation is correct!")

if __name__ == '__main__':
  test_gpt2('gpt2')
