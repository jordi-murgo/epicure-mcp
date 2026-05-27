#!/usr/bin/env bash
#
# One-time Azure resource provisioning for the public Epicure MCP server.
#
# Creates:
#   - Resource group (RG)
#   - Azure Container Registry (ACR, Basic SKU)
#   - Container Apps environment with Log Analytics workspace
#   - First container app revision (scale-to-zero, max 3 replicas)
#   - Microsoft Entra app + federated credential for GitHub Actions OIDC
#
# Prerequisites:
#   az login
#   az account set --subscription "<sub-id>"
#
# Env vars you can override:
#   RG              resource group name           default: epicure-mcp-ne
#   LOCATION        Azure region                  default: northeurope
#   ACR_NAME        ACR name (must be globally unique, lowercase, no dashes)
#   ACA_ENV         Container Apps env name       default: epicure-mcp-env
#   APP_NAME        Container app name            default: epicure-mcp
#   GH_REPO         GitHub repo for OIDC subject  default: KAIKAKU-AI/epicure-mcp

set -euo pipefail

RG="${RG:-epicure-mcp-ne}"
LOCATION="${LOCATION:-northeurope}"
ACR_NAME="${ACR_NAME:-epicuremcpacr$RANDOM}"
ACA_ENV="${ACA_ENV:-epicure-mcp-env}"
APP_NAME="${APP_NAME:-epicure-mcp}"
GH_REPO="${GH_REPO:-KAIKAKU-AI/epicure-mcp}"

SUB_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)

echo "Subscription: $SUB_ID"
echo "Tenant:       $TENANT_ID"
echo "Region:       $LOCATION"
echo "RG:           $RG"
echo "ACR:          $ACR_NAME"
echo "ACA env:      $ACA_ENV"
echo "App:          $APP_NAME"
echo

# Provider registration (idempotent)
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry

echo "[1/5] Resource group"
az group create -n "$RG" -l "$LOCATION" >/dev/null

echo "[2/5] Container Registry ($ACR_NAME)"
az acr create -g "$RG" -n "$ACR_NAME" --sku Basic --admin-enabled true >/dev/null

echo "[3/5] Container Apps environment"
az containerapp env create -g "$RG" -n "$ACA_ENV" -l "$LOCATION" >/dev/null

echo "[4/5] First container app revision (placeholder image)"
ACR_LOGIN_SERVER=$(az acr show -n "$ACR_NAME" -g "$RG" --query loginServer -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$ACR_NAME" --query 'passwords[0].value' -o tsv)
az containerapp create \
    -g "$RG" -n "$APP_NAME" \
    --environment "$ACA_ENV" \
    --image "mcr.microsoft.com/azuredocs/aci-helloworld:latest" \
    --ingress external --target-port 8080 \
    --min-replicas 0 --max-replicas 3 \
    --cpu 0.5 --memory 1.0Gi \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_NAME" \
    --registry-password "$ACR_PASSWORD" >/dev/null

FQDN=$(az containerapp show -g "$RG" -n "$APP_NAME" \
    --query 'properties.configuration.ingress.fqdn' -o tsv)
echo "    -> ingress: https://$FQDN"

echo "[5/5] GitHub Actions OIDC federation"
APP_ID=$(az ad app create --display-name "${APP_NAME}-deploy" --query appId -o tsv)
SP_ID=$(az ad sp create --id "$APP_ID" --query id -o tsv 2>/dev/null || \
        az ad sp show --id "$APP_ID" --query id -o tsv)

az role assignment create \
    --role Contributor \
    --assignee "$APP_ID" \
    --scope "/subscriptions/$SUB_ID/resourceGroups/$RG" >/dev/null || true

az role assignment create \
    --role AcrPush \
    --assignee "$APP_ID" \
    --scope "/subscriptions/$SUB_ID/resourceGroups/$RG/providers/Microsoft.ContainerRegistry/registries/$ACR_NAME" >/dev/null || true

az ad app federated-credential create --id "$APP_ID" --parameters "{
  \"name\":\"github-main\",
  \"issuer\":\"https://token.actions.githubusercontent.com\",
  \"subject\":\"repo:${GH_REPO}:ref:refs/heads/main\",
  \"audiences\":[\"api://AzureADTokenExchange\"]
}" >/dev/null || true

cat <<EOF

Done. Set these as GitHub Actions secrets / variables on $GH_REPO:

  Repository secrets:
    AZURE_CLIENT_ID            = $APP_ID
    AZURE_TENANT_ID            = $TENANT_ID
    AZURE_SUBSCRIPTION_ID      = $SUB_ID

  Repository variables:
    ACR_NAME                   = $ACR_NAME

Public MCP endpoint (will serve the placeholder image until the GitHub
Actions deploy.yml runs and replaces it with the real container):
  https://$FQDN/mcp
  https://$FQDN/healthz
EOF
