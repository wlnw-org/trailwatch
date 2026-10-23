#!/bin/bash
# Launch TrailWatch on AWS Graviton from AWS CloudShell (or any shell with the AWS CLI).
# Creates a security group (HTTP only), an arm64 Amazon Linux 2023 t4g.medium instance with
# deploy/ec2-user-data.sh, and an Elastic IP so the endpoint address stays the same.
set -euo pipefail
REGION="${REGION:-ap-south-1}"
TYPE="${TYPE:-t4g.medium}"
NAME=trailwatch
export AWS_DEFAULT_REGION="$REGION"

AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
      --query Parameter.Value --output text)
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$NAME-http Name=vpc-id,Values="$VPC" \
     --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name $NAME-http --description "TrailWatch HTTP" --vpc-id "$VPC" \
       --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr 0.0.0.0/0
fi
curl -fsSL https://raw.githubusercontent.com/wlnw-org/trailwatch/main/deploy/ec2-user-data.sh -o /tmp/tw-user-data.sh
ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --security-group-ids "$SG" \
     --user-data file:///tmp/tw-user-data.sh --metadata-options HttpTokens=required \
     --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
     --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
     --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$ID"
ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
aws ec2 associate-address --instance-id "$ID" --allocation-id "$ALLOC" >/dev/null
IP=$(aws ec2 describe-addresses --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)
echo "instance=$ID ip=$IP"
echo "The image builds on first boot (about 10 to 15 minutes). Then open: http://$IP/"
