import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import flask
import json
from flask import request

tokenizer, model = None, None
app = flask.Flask(__name__)


def load_model():
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

    return tokenizer, model
tokenizer, model = load_model()


{
    'note': 'info or discription, anything ',
    'messages': [
        {
            'role': "システム",
            'content': 'こんにちは、りんなです。'
        },
        {
            'role': "ユーザー",
            'content': 'こんにちは、りんなです。'
        }
    ]
}


def generate_prompt(messages: dict) -> str:
    prompt = [
        f"{uttr['role']}: {uttr['content']}"
        for uttr in messages
    ]

    prompt = "<NL>".join(prompt)
    prompt = (
        prompt
        + "<NL>"
        + "システム: "
    )

    return prompt


def generate_reply(prompt: str) -> str:
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

    output = tokenizer.decode(output_ids.tolist()[token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    return output


@app.route('/get_reply', methods=["post"])
def get_reply():
    data = request.data.decode('utf-8')
    data = json.loads(data)
    # print(data)
    prompt = generate_prompt(data["messages"])
    reply = generate_reply(prompt)
    data['messages'].append({'role':'システム', 'content':reply})
    return flask.jsonify(data)


application = app
if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0')
