import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

prompt = [
    {
        "speaker": "ユーザー",
        "text": "コンタクトレンズを慣れるにはどうすればよいですか？"
    },
    {
        "speaker": "システム",
        "text": "これについて具体的に説明していただけますか？何が難しいのでしょうか？"
    },
    {
        "speaker": "ユーザー",
        "text": "目が痛いのです。"
    },
    {
        "speaker": "システム",
        "text": "分かりました、コンタクトレンズをつけると目がかゆくなるということですね。思った以上にレンズを外す必要があるでしょうか？"
    },
    {
        "speaker": "ユーザー",
        "text": "いえ、レンズは外しませんが、目が赤くなるんです。"
    }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
]

print('\n'.join(prompt))

prompt = "<NL>".join(prompt)
prompt = (
    prompt
    + "<NL>"
    + "システム: "
)




bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    "rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False, legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

# if torch.cuda.is_available():
#     model = model.to("cuda")

token_ids = tokenizer.encode(
    prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        do_sample=True,
        max_new_tokens=128,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output_ids = output_ids[output_ids != tokenizer.pad_token_id]
output_ids = output_ids[output_ids != tokenizer.bos_token_id]
output_ids = output_ids[output_ids != tokenizer.eos_token_id]
# print(output_ids)
# print(output_ids.tolist())
output = tokenizer.decode(output_ids.tolist()[token_ids.size(1):])
output = output.replace("<NL>", "\n")
print('システム:', output)
"""それは、コンタクトレンズが目に合わないために起こることがあります。レンズが目の表面に長時間触れ続けることが原因となることがあります。また、コンタクトレンズが汚れている可能性もあります。コンタクトレンズケースを定期的に洗浄したり、コンタクトレンズを正しくフィットさせるようにしたりすることが役立ちます。</s>"""
