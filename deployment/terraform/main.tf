terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
  }
}

variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository in OWNER/REPO format for CI/CD integration"
  type        = string
  default     = "adrinee/haiku-agent"
}

variable "region" {
  description = "Google Cloud Region for Vertex AI Agent Engine"
  type        = string
  default     = "europe-west1"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required Google Cloud APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudtrace.googleapis.com",
    "dlp.googleapis.com",
    "run.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# Dedicated least-privilege Service Account for Haiku Agent
resource "google_service_account" "haiku_agent_sa" {
  account_id   = "haiku-agent-runtime-sa"
  display_name = "Haiku Agent Runtime Service Account (${var.github_repo})"
  depends_on   = [google_project_service.required_apis]
}

# Least-privilege IAM roles (Vertex AI, Secret Manager, Cloud Trace, Cloud DLP)
resource "google_project_iam_member" "haiku_agent_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/cloudtrace.agent",
    "roles/dlp.user",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.haiku_agent_sa.email}"
}

# Managed Secret container for Google Chat Webhook (no secret value in Terraform code)
resource "google_secret_manager_secret" "haiku_chat_webhook" {
  secret_id = "haiku-chat-webhook"
  labels = {
    managed_by = "terraform"
  }
  replication {
    auto {}
  }
  depends_on = [google_project_service.required_apis]
}

output "service_account_email" {
  description = "Dedicated Service Account email for Haiku Agent Runtime"
  value       = google_service_account.haiku_agent_sa.email
}

output "chat_webhook_secret_id" {
  description = "Secret Manager ID for the Google Chat webhook"
  value       = google_secret_manager_secret.haiku_chat_webhook.secret_id
}
