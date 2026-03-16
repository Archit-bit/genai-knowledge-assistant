# Azure App Service Deployment

This app can be deployed to Azure App Service on Linux and exposed at:

`https://<your-app-name>.azurewebsites.net`

## Recommended public deployment mode

For a resume link, use one of these:

- `EMBEDDING_BACKEND=hashing` and `GENERATION_BACKEND=extractive` for guaranteed zero model cost.
- `EMBEDDING_BACKEND=hashing` and `GENERATION_BACKEND=gemini` for free-tier Gemini generation.

Using `hashing` for retrieval keeps embeddings local, simple, and cheap. If you use Gemini on the public app, quota still applies.

## 1. Create Azure resources

```bash
APP_NAME="genai-knowledge-assistant-demo"
RESOURCE_GROUP="genai-knowledge-assistant-rg"
PLAN_NAME="genai-knowledge-assistant-plan"
LOCATION="eastus"

az login
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
az appservice plan create \
  --name "$PLAN_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --sku B1 \
  --is-linux
az webapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$PLAN_NAME" \
  --runtime "PYTHON|3.10"
```

Notes:

- `B1` is the safer tier if you want a reliable public resume link.
- If you only need the default `azurewebsites.net` URL, you can stay with that hostname.
- Custom domains require a paid App Service tier.

## 2. Configure build and startup

```bash
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true

az webapp config set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --startup-file "bash startup.sh"
```

## 3. Configure app settings

Zero-cost mode:

```bash
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    EMBEDDING_BACKEND=hashing \
    GENERATION_BACKEND=extractive \
    TOP_K=4 \
    STRICT_GROUNDING=true
```

Gemini free-tier mode:

```bash
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    GEMINI_API_KEY="<your-gemini-api-key>" \
    EMBEDDING_BACKEND=hashing \
    GENERATION_BACKEND=gemini \
    GENERATION_MODEL=gemini-2.5-flash \
    TOP_K=4 \
    STRICT_GROUNDING=true
```

## 4. Deploy code

From the project root:

```bash
zip -r deploy.zip . \
  -x ".git/*" \
  -x ".venv/*" \
  -x "__pycache__/*" \
  -x ".pytest_cache/*" \
  -x "data/index/*"

az webapp deploy \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --src-path deploy.zip \
  --type zip
```

## 5. Open the app

```bash
echo "https://${APP_NAME}.azurewebsites.net"
```

## Notes for a resume link

- Public apps can receive arbitrary user prompts, so do not expose paid model settings unless you are comfortable with the quota or billing behavior.
- Azure App Service instance storage is ephemeral. Uploaded files and generated indexes are fine for demo use, but they are not durable storage.
- If you want your own domain later, bind it after the app is live on a paid tier.
