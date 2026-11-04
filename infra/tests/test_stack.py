"""Synthesize the stack and check the important resources and settings."""

import sys
from pathlib import Path

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk.assertions import Match, Template  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stack import LAMBDA_ZIP, ForgeGptStack  # noqa: E402

pytestmark = pytest.mark.skipif(not LAMBDA_ZIP.exists(), reason="build the Lambda zip first")


def synth(**context) -> Template:
    app = cdk.App(context={"api_key": "test", **context})
    return Template.from_stack(ForgeGptStack(app, "Test", env=cdk.Environment(region="us-east-1")))


def test_lambda_and_function_url():
    t = synth()
    t.has_resource_properties(
        "AWS::Lambda::Function",
        {"Runtime": "python3.12", "MemorySize": 2048, "Timeout": 30,
         "Environment": {"Variables": Match.object_like({"EMF": "1", "API_KEY": "test"})}},
    )  # fmt: skip
    t.resource_count_is("AWS::Lambda::Url", 2)
    t.has_resource_properties("AWS::S3::Bucket", {"VersioningConfiguration": {"Status": "Enabled"}})
    t.resource_count_is("AWS::EC2::Instance", 0)  # EC2 is opt-in
    t.resource_count_is("AWS::CloudWatch::Dashboard", 1)


def test_ec2_is_free_plan_eligible_and_locked_down():
    t = synth(ec2="true", my_ip="203.0.113.7/32", budget_email="me@example.com")
    t.has_resource_properties("AWS::EC2::Instance", {"InstanceType": "m7i-flex.large"})
    t.resource_count_is("AWS::EC2::NatGateway", 0)
    t.has_resource_properties(
        "AWS::EC2::SecurityGroup",
        {
            "SecurityGroupIngress": [
                Match.object_like({"CidrIp": "203.0.113.7/32", "FromPort": 8080})
            ]
        },
    )
    t.resource_count_is("AWS::Budgets::Budget", 1)


def test_ec2_requires_ip():
    with pytest.raises(ValueError):
        synth(ec2="true")
