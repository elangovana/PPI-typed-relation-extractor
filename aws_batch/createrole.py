import json
import logging

import boto3


def create_role(role_name, assumed_role_policy, policy, managed_policy_arns=None):
    logger = logging.getLogger(__name__)
    client = boto3.client('iam')
    managed_policy_arns = managed_policy_arns or []

    #TODO: Fix this update role later
    try:
        response = client.create_role(

            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps( assumed_role_policy),
            Description='This is the role for a batch task'
        )
    except Exception as e:
        logger.warning("Error creating role {}, {}".format(role_name, e))



    for p in managed_policy_arns:
        response = client.attach_role_policy(
            RoleName=role_name, PolicyArn=p)

    role_policy = boto3.resource('iam').RolePolicy(role_name, 'role_custom_policy')
    response = role_policy.put(
        PolicyDocument=json.dumps(policy)
    )