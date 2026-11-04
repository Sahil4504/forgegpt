#!/usr/bin/env python3
"""CDK app for ForgeGPT. Deploy / tear down everything with one command:

    cd infra
    cdk deploy -c api_key=$FORGEGPT_API_KEY                    # S3 + Lambda + dashboard
    cdk deploy -c api_key=... -c ec2=true -c my_ip=1.2.3.4/32  # + llama.cpp EC2 box
    cdk destroy                                                # remove everything

Build the Lambda zip first: python serving/lambda/build.py
"""

import os

import aws_cdk as cdk
from stack import ForgeGptStack

app = cdk.App()
ForgeGptStack(
    app,
    "ForgeGpt",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
)
app.synth()
