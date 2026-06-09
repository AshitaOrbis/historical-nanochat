#!/usr/bin/env python3
"""Simple OpenAI-compatible API server for Historical Nanochat."""

import sys
sys.path.insert(0, "/home/user/historical-nanochat/nanochat")

import torch
import json
import time
import uuid
from typing import Optional, List
from dataclasses import dataclass
from flask import Flask, request, jsonify, Response

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.tokenizer import get_tokenizer

app = Flask(__name__)

# Global model and tokenizer
model = None
tokenizer = None
device = None
config = None

def load_model():
    global model, tokenizer, device, config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    checkpoint_dir = "/home/user/.cache/nanochat/base_checkpoints/d12_v1"
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, 15250, device, load_optimizer=False)

    cfg = meta_data["model_config"]
    config = GPTConfig(**cfg)
    model = GPT(config).to(device)
    model.load_state_dict(model_data, strict=True)
    model.eval()

    tokenizer = get_tokenizer()
    print("Model loaded successfully!")

@torch.no_grad()
def generate(prompt: str, max_tokens: int = 100, temperature: float = 0.8,
             top_k: int = 40, stop: Optional[List[str]] = None) -> str:
    """Generate text from prompt."""
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    generated_text = ""

    for _ in range(max_tokens):
        if tokens.size(1) >= config.sequence_len:
            break

        logits = model(tokens)[:, -1, :] / temperature

        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        # Decode just the new token
        new_text = tokenizer.decode([next_token.item()])
        generated_text += new_text

        # Check stop sequences
        if stop:
            for s in stop:
                if s in generated_text:
                    generated_text = generated_text.split(s)[0]
                    return generated_text

    return generated_text

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "historical-nanochat-125m",
            "object": "model",
            "created": 1700000000,
            "owned_by": "local"
        }]
    })

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completions endpoint."""
    data = request.json

    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 40)
    stop = data.get('stop', None)
    stream = data.get('stream', False)

    if stream:
        def generate_stream():
            tokens = tokenizer.encode(prompt)
            tokens_t = torch.tensor([tokens], dtype=torch.long, device=device)

            for i in range(max_tokens):
                if tokens_t.size(1) >= config.sequence_len:
                    break

                with torch.no_grad():
                    logits = model(tokens_t)[:, -1, :] / temperature
                    if top_k:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float('-inf')
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                tokens_t = torch.cat([tokens_t, next_token], dim=1)
                new_text = tokenizer.decode([next_token.item()])

                chunk = {
                    "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "historical-nanochat-125m",
                    "choices": [{
                        "text": new_text,
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return Response(generate_stream(), mimetype='text/event-stream')

    generated = generate(prompt, max_tokens, temperature, top_k, stop)

    return jsonify({
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "historical-nanochat-125m",
        "choices": [{
            "text": generated,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(generated)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(generated))
        }
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.json

    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.8)

    # Convert chat messages to prompt
    prompt = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            prompt += f"{content}\n\n"
        elif role == 'user':
            prompt += f"{content}\n"
        elif role == 'assistant':
            prompt += f"{content}\n"

    generated = generate(prompt, max_tokens, temperature)

    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "historical-nanochat-125m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated
            },
            "finish_reason": "stop"
        }]
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    load_model()
    print("\n" + "="*50)
    print("Historical Nanochat API Server")
    print("="*50)
    print("Endpoints:")
    print("  GET  /v1/models")
    print("  POST /v1/completions")
    print("  POST /v1/chat/completions")
    print("  GET  /health")
    print("\nExample usage:")
    print('  curl http://localhost:5000/v1/completions \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"prompt": "In the year 1850,", "max_tokens": 50}\'')
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=5000)
