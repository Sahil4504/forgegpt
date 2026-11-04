# Run it yourself: Kaggle training and AWS deployment

Everything below uses your own accounts. Budget: about $10–20 of AWS Free plan credits
in total, $0 on Kaggle. Region: `us-east-1` throughout.

## 0. One-time AWS setup (console, ~30 min)

1. Create the account on the **Free plan** (not Paid). Note the creation date: the plan
   ends 6 months later.
2. Turn on MFA for the root user. Create an IAM user with `AdministratorAccess` for daily
   work and never create root access keys.
3. **Billing → Budgets**: create a cost budget with email alerts at $5, $20 and $50
   (or pass `-c budget_email=you@example.com` to the CDK deploy below).
4. Earn the extra $100 of credits with the five onboarding activities (launch an EC2
   instance, an RDS database, a Lambda function with a Function URL, a Bedrock prompt,
   an AWS Budget). Delete the EC2 and RDS resources straight afterwards.
5. On your laptop: install the AWS CLI and run `aws configure --profile forgegpt`
   (access key of the IAM user, region `us-east-1`). Then `export AWS_PROFILE=forgegpt`
   (PowerShell: `$env:AWS_PROFILE="forgegpt"`).
6. Install Node.js 20+ and the CDK CLI: `npm install -g aws-cdk`, then once per
   account/region: `cdk bootstrap`.

## 1. Full training run on Kaggle (free GPU, ~1 hour)

1. Create a **private Kaggle dataset** called `forgegpt-tokens-v1` with
   `data/tokens/v1/{train,val,test}.bin`, `data/tokens/v1/meta.json` and
   `data/tokenizer/v1/bpe-8k.json` (~330 MB).
2. Push this repo to GitHub, upload `notebooks/kaggle_train.ipynb` to Kaggle, set
   `REPO_URL`, choose a T4 GPU, add the dataset, run all cells.
3. Download `runs/t4_6L384/` from the notebook's Output tab into `runs/t4_6L384/` locally.
4. Re-run the local steps against the new checkpoint:

```bash
python -m forgegpt.onnx_export --ckpt runs/t4_6L384/ckpt_best.pt --out-dir artifacts/onnx
python scripts/benchmark_inference.py --ckpt runs/t4_6L384/ckpt_best.pt
python scripts/make_charts.py --run runs/t4_6L384
```

## 2. Deploy (one command)

```bash
python serving/lambda/build.py                       # -> serving/lambda/dist/forgegpt-lambda.zip
cd infra
pip install -r requirements.txt
export FORGEGPT_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
cdk deploy -c api_key=$FORGEGPT_API_KEY              # S3 + 2 Lambdas + dashboard
```

The output prints `FunctionUrl`. Test it:

```bash
curl -s -X POST "$FUNCTION_URL" -H "x-api-key: $FORGEGPT_API_KEY" \
  -H "content-type: application/json" \
  -d '{"prompt": "the brakes", "max_tokens": 60, "seed": 1}'
```

Add the llama.cpp server (costs ~$0.10/hour while running):

```bash
cdk deploy -c api_key=$FORGEGPT_API_KEY -c ec2=true -c my_ip=$(curl -s https://checkip.amazonaws.com)/32
```

The instance builds llama.cpp and downloads both GGUF files on first boot (~10 min).
Check progress with Session Manager (EC2 console → Connect → Session Manager):
`sudo tail -f /var/log/forgegpt-setup.log`.

## 3. Benchmarks on AWS

```bash
# Lambda: cold and warm latency, then load
python scripts/run_load_test.py --host "$FUNCTION_URL" --target lambda --label aws_lambda
# llama.cpp on EC2, Q4_K_M (default), then Q8_0
python scripts/run_load_test.py --host "http://<LlamaUrl>" --target llamacpp --label llamacpp_q4_k_m
#   on the instance: sudo bash /path/to/bench_llamacpp.sh use q8_0
python scripts/run_load_test.py --host "http://<LlamaUrl>" --target llamacpp --label llamacpp_q8_0
python scripts/cost_model.py && python scripts/make_charts.py
```

The CloudWatch dashboard **ForgeGPT** shows Lambda p50/p95 duration, errors, time to
first token and tokens/s (from the EMF log lines, no extra API calls).

## 4. Stop and tear down

- After every EC2 session: `aws ec2 stop-instances --instance-ids <LlamaInstanceId>`.
- When finished: `cd infra && cdk destroy`. This deletes the bucket (and its objects),
  both Lambdas, log groups, the dashboard, the VPC and the instance.
- Check **Billing → Free Tier and Credits** weekly, and release any Elastic IPs or
  leftover EBS volumes (there should be none: the stack uses neither).
