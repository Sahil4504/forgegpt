"""Locust load test for ForgeGPT endpoints.

    TARGET=forgegpt locust -f benchmarks/locustfile.py --host http://localhost:8000
    TARGET=lambda   locust -f benchmarks/locustfile.py --host https://<id>.lambda-url.us-east-1.on.aws
    TARGET=llamacpp locust -f benchmarks/locustfile.py --host http://<ec2-ip>:8080

Env: API_KEY (sent as x-api-key / Bearer), PROMPT=short|long, MAX_TOKENS (default 64).
Usually driven by scripts/run_load_test.py, which sweeps the number of users.
"""

import os
import random

from locust import HttpUser, between, task

TARGET = os.environ.get("TARGET", "forgegpt")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
API_KEY = os.environ.get("API_KEY", "")
SHORT = ["the brakes", "the airbag light", "while driving the", "the transmission", "the dealer"]
LONG = [
    "i was driving on the interstate at about 70 mph in moderate traffic when the vehicle "
    "suddenly lost power. the dashboard lit up with several warning lights including the "
    "check engine light and the traction control light. i pulled over safely and",
    "the contact owns a 2019 honda cr-v. the contact stated that while driving at various "
    "speeds the vehicle hesitated and the check engine warning light illuminated. the "
    "vehicle was taken to the dealer where it was diagnosed that",
]
PROMPTS = LONG if os.environ.get("PROMPT") == "long" else SHORT


class ForgeUser(HttpUser):
    wait_time = between(0.2, 0.5)

    @task
    def generate(self):
        prompt = random.choice(PROMPTS)
        if TARGET == "llamacpp":
            self.client.post(
                "/v1/chat/completions",
                name="llamacpp chat",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "messages": [
                        {"role": "system", "content": "Continue the vehicle complaint."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.8,
                },
            )
        else:
            path = "/" if TARGET == "lambda" else "/v1/generate"
            self.client.post(
                path,
                name=f"{TARGET} generate",
                headers={"x-api-key": API_KEY},
                json={"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0.8},
            )
