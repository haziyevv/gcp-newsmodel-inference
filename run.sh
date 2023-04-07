gcloud builds submit --config cloudbuild.yaml

!gcloud ai models upload \
  --container-ports=80 \
  --container-predict-route="/predict" \
  --container-health-route="/health" \
  --region=us-central1 \
  --display-name=azeds-fast-api \
  --container-image-uri=gcr.io/test-azer-ds/aze-ds-vertex-predict