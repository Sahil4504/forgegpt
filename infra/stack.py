"""ForgeGPT infrastructure, sized for the AWS Free plan (credits, no Paid plan).

Always created (cost ~$0 while idle):
  - S3 bucket for model artifacts (versioned, private, deleted on `cdk destroy`)
  - Lambda (Python 3.12, x86_64) + Function URL serving the INT8 ONNX model,
    and a second function that loads the same model from S3 at cold start
  - CloudWatch log groups (1 week retention) and one dashboard
  - Optional AWS Budget alerts at $5 / $20 / $50 (context budget_email)
Only with -c ec2=true (costs ~$0.10/hour while running, stop it after each session):
  - m7i-flex.large in a public subnet of a new VPC (no NAT gateway), reachable only
    from my_ip on port 8080, managed via SSM Session Manager (no SSH keys)
"""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_ZIP = ROOT / "serving" / "lambda" / "dist" / "forgegpt-lambda.zip"
ONNX_DIR = ROOT / "artifacts" / "onnx"


class ForgeGptStack(cdk.Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)
        ctx = self.node.try_get_context
        api_key = ctx("api_key") or ""
        if not api_key:
            cdk.Annotations.of(self).add_warning(
                "No api_key context set: the Function URL will accept any request."
            )

        # --- storage ------------------------------------------------------------------
        bucket = s3.Bucket(
            self,
            "Artifacts",
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        if ONNX_DIR.exists():
            s3deploy.BucketDeployment(
                self,
                "UploadModel",
                sources=[s3deploy.Source.asset(str(ONNX_DIR))],
                destination_bucket=bucket,
                destination_key_prefix="models/forgegpt/v1",
            )

        # --- serverless serving -----------------------------------------------------------
        common_env = {
            "API_KEY": api_key,
            "EMF": "1",
            "MAX_TOKENS_CAP": str(ctx("max_tokens_cap") or 200),
            "ORT_THREADS": "2",
        }
        memory = int(ctx("lambda_memory_mb") or 2048)

        def make_function(name: str, extra_env: dict) -> lambda_.Function:
            log_group = logs.LogGroup(
                self,
                f"{name}Logs",
                retention=logs.RetentionDays.ONE_WEEK,
                removal_policy=cdk.RemovalPolicy.DESTROY,
            )
            return lambda_.Function(
                self,
                name,
                runtime=lambda_.Runtime.PYTHON_3_12,
                architecture=lambda_.Architecture.X86_64,
                handler="handler.handler",
                code=lambda_.Code.from_asset(str(LAMBDA_ZIP)),
                memory_size=memory,
                timeout=cdk.Duration.seconds(30),
                environment={**common_env, **extra_env},
                log_group=log_group,
            )

        fn = make_function("Generate", {})
        fn_s3 = make_function(
            "GenerateFromS3",
            {"MODEL_S3_URI": f"s3://{bucket.bucket_name}/models/forgegpt/v1/forgegpt.int8.onnx"},
        )
        bucket.grant_read(fn_s3)

        urls = {}
        for f, label in ((fn, "FunctionUrl"), (fn_s3, "FunctionUrlS3")):
            url = f.add_function_url(
                auth_type=lambda_.FunctionUrlAuthType.NONE,  # API key is checked in code
                cors=lambda_.FunctionUrlCorsOptions(
                    allowed_origins=["*"], allowed_methods=[lambda_.HttpMethod.POST]
                ),
            )
            urls[label] = url
            cdk.CfnOutput(self, label, value=url.url)

        # --- open-model serving on EC2 (optional) -------------------------------------------
        instance = None
        if str(ctx("ec2")).lower() == "true":
            my_ip = ctx("my_ip")
            if not my_ip:
                raise ValueError("-c ec2=true needs -c my_ip=<your.ip.address>/32")
            vpc = ec2.Vpc(
                self,
                "Vpc",
                max_azs=1,
                nat_gateways=0,  # NAT gateways are expensive and not needed here
                subnet_configuration=[
                    ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC)
                ],
            )
            sg = ec2.SecurityGroup(self, "LlamaSg", vpc=vpc, allow_all_outbound=True)
            sg.add_ingress_rule(ec2.Peer.ipv4(my_ip), ec2.Port.tcp(8080), "llama.cpp API")
            role = iam.Role(self, "LlamaRole", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"))
            role.add_managed_policy(
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
            )
            script = (ROOT / "serving" / "ec2" / "user_data.sh").read_text()
            script = script.replace("__API_KEY__", api_key).replace(
                "__QUANT__", ctx("ec2_quant") or "q4_k_m"
            )
            instance = ec2.Instance(
                self,
                "Llama",
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                instance_type=ec2.InstanceType("m7i-flex.large"),
                machine_image=ec2.MachineImage.from_ssm_parameter(
                    "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
                ),
                security_group=sg,
                role=role,
                user_data=ec2.UserData.custom(script),
                block_devices=[
                    ec2.BlockDevice(
                        device_name="/dev/sda1",
                        volume=ec2.BlockDeviceVolume.ebs(
                            20, volume_type=ec2.EbsDeviceVolumeType.GP3, delete_on_termination=True
                        ),
                    )
                ],
                require_imdsv2=True,
            )
            cdk.CfnOutput(self, "LlamaUrl", value=f"http://{instance.instance_public_ip}:8080")
            cdk.CfnOutput(self, "LlamaInstanceId", value=instance.instance_id)

        # --- observability --------------------------------------------------------------------
        dash = cw.Dashboard(self, "Dashboard", dashboard_name="ForgeGPT")

        def emf(name: str, stat: str, label: str) -> cw.Metric:
            return cw.Metric(
                namespace="ForgeGPT",
                metric_name=name,
                dimensions_map={"service": "lambda"},
                statistic=stat,
                label=label,
                period=cdk.Duration.minutes(1),
            )

        dash.add_widgets(
            cw.GraphWidget(
                title="Lambda duration (ms)",
                left=[
                    fn.metric_duration(statistic="p50", label="p50"),
                    fn.metric_duration(statistic="p95", label="p95"),
                ],
            ),
            cw.GraphWidget(
                title="Requests and errors",
                left=[fn.metric_invocations(), fn.metric_errors(), fn.metric_throttles()],
            ),
            cw.GraphWidget(
                title="Time to first token (ms)",
                left=[emf("ttft_ms", "p50", "p50"), emf("ttft_ms", "p95", "p95")],
            ),
            cw.GraphWidget(
                title="Generation speed (tokens/s)",
                left=[emf("tokens_per_s", "Average", "mean")],
            ),
        )
        if instance is not None:
            dash.add_widgets(
                cw.GraphWidget(
                    title="EC2 CPU (%)",
                    left=[
                        cw.Metric(
                            namespace="AWS/EC2",
                            metric_name="CPUUtilization",
                            dimensions_map={"InstanceId": instance.instance_id},
                        )
                    ],
                )
            )
        cdk.CfnOutput(self, "BucketName", value=bucket.bucket_name)

        # --- cost guardrail --------------------------------------------------------------------
        email = ctx("budget_email")
        if email:
            budgets.CfnBudget(
                self,
                "Budget",
                budget=budgets.CfnBudget.BudgetDataProperty(
                    budget_type="COST",
                    time_unit="MONTHLY",
                    budget_limit=budgets.CfnBudget.SpendProperty(amount=50, unit="USD"),
                ),
                notifications_with_subscribers=[
                    budgets.CfnBudget.NotificationWithSubscribersProperty(
                        notification=budgets.CfnBudget.NotificationProperty(
                            comparison_operator="GREATER_THAN",
                            notification_type="ACTUAL",
                            threshold=amount,
                            threshold_type="ABSOLUTE_VALUE",
                        ),
                        subscribers=[
                            budgets.CfnBudget.SubscriberProperty(
                                address=email, subscription_type="EMAIL"
                            )
                        ],
                    )
                    for amount in (5, 20, 50)
                ],
            )
