"""Generate the synthetic employees + learning-content datasets used by the demo.

Run once to (re)create:
    data/learning_content.csv
    data/employees.csv

The output is deterministic (fixed seed) so reviewers can reproduce results.
No real personal or proprietary data is involved.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Learning content catalogue
# ---------------------------------------------------------------------------

CONTENT: list[dict] = [
    # --- Data engineering / SQL ---
    {
        "name": "Advanced SQL for Analytics",
        "description": "Window functions, CTEs, query optimization, and warehouse-specific tips for Snowflake and BigQuery.",
        "keywords": ["sql", "analytics", "warehouse", "optimization"],
        "skills": ["SQL", "Data Analysis", "BigQuery", "Snowflake"],
        "duration": 5400,
    },
    {
        "name": "Building Production ETL with Airflow",
        "description": "Author idempotent DAGs, manage backfills, and monitor pipeline reliability in Apache Airflow.",
        "keywords": ["airflow", "etl", "orchestration", "data engineering"],
        "skills": ["Airflow", "Python", "ETL", "Data Engineering"],
        "duration": 7200,
    },
    {
        "name": "Data Modeling with dbt",
        "description": "Structure analytics codebases with dbt: staging/marts layering, tests, snapshots, and docs.",
        "keywords": ["dbt", "modeling", "analytics engineering"],
        "skills": ["dbt", "SQL", "Data Modeling"],
        "duration": 4800,
    },
    {
        "name": "Streaming Data with Kafka",
        "description": "Produce, consume, and process event streams; partitioning, consumer groups, and exactly-once semantics.",
        "keywords": ["kafka", "streaming", "events"],
        "skills": ["Kafka", "Streaming", "Distributed Systems"],
        "duration": 6000,
    },
    {
        "name": "Spark for Big Data Processing",
        "description": "PySpark fundamentals: DataFrame API, partitioning, joins, and tuning for large-scale jobs.",
        "keywords": ["spark", "pyspark", "big data"],
        "skills": ["Spark", "Python", "Big Data"],
        "duration": 7200,
    },
    # --- Cloud / DevOps ---
    {
        "name": "AWS Cloud Practitioner Essentials",
        "description": "Core AWS services overview: EC2, S3, IAM, VPC, billing. Foundation for the Cloud Practitioner exam.",
        "keywords": ["aws", "cloud", "certification"],
        "skills": ["AWS", "Cloud Computing"],
        "duration": 9000,
    },
    {
        "name": "Kubernetes in Production",
        "description": "Workload patterns, ingress, autoscaling, observability, and zero-downtime deployments.",
        "keywords": ["kubernetes", "k8s", "devops", "containers"],
        "skills": ["Kubernetes", "DevOps", "Docker"],
        "duration": 8400,
    },
    {
        "name": "Terraform for Infrastructure as Code",
        "description": "Provision multi-cloud infrastructure with reusable modules, remote state, and CI integration.",
        "keywords": ["terraform", "iac", "devops"],
        "skills": ["Terraform", "IaC", "DevOps"],
        "duration": 5400,
    },
    {
        "name": "CI/CD with GitHub Actions",
        "description": "Build, test, and ship pipelines: matrix jobs, reusable workflows, secrets, and environment gating.",
        "keywords": ["ci", "cd", "github actions", "automation"],
        "skills": ["CI/CD", "GitHub Actions", "DevOps"],
        "duration": 3600,
    },
    {
        "name": "Observability with OpenTelemetry",
        "description": "Instrument applications for traces, metrics, and logs; integrate with Prometheus, Grafana, and Jaeger.",
        "keywords": ["observability", "tracing", "monitoring"],
        "skills": ["OpenTelemetry", "Observability", "SRE"],
        "duration": 4800,
    },
    # --- Machine learning / AI ---
    {
        "name": "Practical Machine Learning with scikit-learn",
        "description": "End-to-end ML: feature engineering, model selection, cross-validation, and deployment-ready pipelines.",
        "keywords": ["machine learning", "ml", "scikit-learn"],
        "skills": ["Machine Learning", "Python", "scikit-learn"],
        "duration": 9000,
    },
    {
        "name": "Deep Learning Fundamentals with PyTorch",
        "description": "Tensors, autograd, training loops, CNNs, and transformers in PyTorch.",
        "keywords": ["deep learning", "pytorch", "neural networks"],
        "skills": ["Deep Learning", "PyTorch", "Python"],
        "duration": 10800,
    },
    {
        "name": "MLOps: Shipping ML Models to Production",
        "description": "Experiment tracking, model registries, online/offline serving, and monitoring for drift.",
        "keywords": ["mlops", "deployment", "monitoring"],
        "skills": ["MLOps", "Machine Learning", "DevOps"],
        "duration": 7200,
    },
    {
        "name": "LLMs and Retrieval-Augmented Generation",
        "description": "Build RAG systems: embeddings, vector stores, prompt design, and evaluation methods.",
        "keywords": ["llm", "rag", "embeddings", "vector search"],
        "skills": ["LLMs", "RAG", "NLP", "Python"],
        "duration": 6000,
    },
    {
        "name": "Prompt Engineering for Developers",
        "description": "Patterns for reliable LLM outputs: few-shot prompting, function/tool calling, JSON-mode constraints.",
        "keywords": ["prompt engineering", "llm", "ai"],
        "skills": ["Prompt Engineering", "LLMs"],
        "duration": 3000,
    },
    # --- Software engineering ---
    {
        "name": "Clean Code Principles",
        "description": "Naming, function design, comments, error handling — code that reads like prose.",
        "keywords": ["clean code", "software craftsmanship"],
        "skills": ["Software Engineering", "Code Quality"],
        "duration": 4200,
    },
    {
        "name": "Test-Driven Development in Python",
        "description": "Red-green-refactor with pytest; mocking, fixtures, and property-based tests.",
        "keywords": ["testing", "tdd", "pytest"],
        "skills": ["Testing", "Python", "TDD"],
        "duration": 4800,
    },
    {
        "name": "Designing RESTful APIs",
        "description": "Resource modeling, versioning, idempotency, pagination, and API documentation with OpenAPI.",
        "keywords": ["api", "rest", "design"],
        "skills": ["API Design", "Backend Development"],
        "duration": 4200,
    },
    {
        "name": "System Design Interview Prep",
        "description": "Approach scalable-system questions: capacity estimation, data partitioning, caching, queues.",
        "keywords": ["system design", "scalability", "interview"],
        "skills": ["System Design", "Distributed Systems"],
        "duration": 7200,
    },
    {
        "name": "Modern TypeScript",
        "description": "Generics, conditional types, narrowing, and patterns for typed React components.",
        "keywords": ["typescript", "frontend", "types"],
        "skills": ["TypeScript", "Frontend Development"],
        "duration": 5400,
    },
    {
        "name": "React with Hooks and Suspense",
        "description": "Functional components, custom hooks, server components, and data fetching patterns.",
        "keywords": ["react", "frontend", "hooks"],
        "skills": ["React", "JavaScript", "Frontend Development"],
        "duration": 6000,
    },
    # --- Product / design / leadership ---
    {
        "name": "Product Management Foundations",
        "description": "From discovery to launch: customer interviews, roadmaps, OKRs, and prioritization frameworks.",
        "keywords": ["product management", "roadmap", "okrs"],
        "skills": ["Product Management", "Strategy"],
        "duration": 6000,
    },
    {
        "name": "User Research Methods",
        "description": "Interview techniques, usability studies, surveys, and synthesizing qualitative insights.",
        "keywords": ["user research", "ux", "qualitative"],
        "skills": ["User Research", "UX"],
        "duration": 4800,
    },
    {
        "name": "Figma for Product Designers",
        "description": "Components, auto-layout, variables, prototyping, and design-system maintenance.",
        "keywords": ["figma", "design", "ux"],
        "skills": ["Figma", "UX Design"],
        "duration": 5400,
    },
    {
        "name": "Leading Engineering Teams",
        "description": "1:1s, performance management, technical strategy, and growing senior engineers.",
        "keywords": ["leadership", "engineering management"],
        "skills": ["Leadership", "Engineering Management"],
        "duration": 7200,
    },
    {
        "name": "Effective Technical Communication",
        "description": "Write design docs, give clear status updates, run productive review meetings.",
        "keywords": ["communication", "writing", "soft skills"],
        "skills": ["Communication", "Technical Writing"],
        "duration": 3600,
    },
    {
        "name": "Negotiation Skills for Professionals",
        "description": "Principled negotiation, BATNA, and managing difficult conversations at work.",
        "keywords": ["negotiation", "communication", "soft skills"],
        "skills": ["Negotiation", "Communication"],
        "duration": 4200,
    },
    {
        "name": "Agile and Scrum Essentials",
        "description": "Ceremonies, roles, estimation, and avoiding common anti-patterns in Scrum teams.",
        "keywords": ["agile", "scrum", "process"],
        "skills": ["Agile", "Scrum"],
        "duration": 3600,
    },
    # --- Security ---
    {
        "name": "Application Security Fundamentals",
        "description": "OWASP Top 10, authentication, secrets management, and dependency hygiene.",
        "keywords": ["security", "owasp", "appsec"],
        "skills": ["Application Security", "OWASP"],
        "duration": 4800,
    },
    {
        "name": "Cloud Security on AWS",
        "description": "IAM least privilege, KMS, VPC isolation, GuardDuty, and incident response basics.",
        "keywords": ["security", "aws", "iam"],
        "skills": ["Cloud Security", "AWS", "IAM"],
        "duration": 5400,
    },
    # --- Languages / general ---
    {
        "name": "Business English for Technical Roles",
        "description": "Email, meetings, and presentations: vocabulary and patterns for international teams.",
        "keywords": ["english", "language", "communication"],
        "skills": ["English", "Communication"],
        "duration": 9000,
    },
    {
        "name": "German for the Workplace (B1)",
        "description": "Practical German for office communication: meetings, emails, and small talk.",
        "keywords": ["german", "language", "communication"],
        "skills": ["German", "Communication"],
        "duration": 18000,
    },
    {
        "name": "Time Management for Knowledge Workers",
        "description": "Calendar blocking, async-first habits, and protecting focus time.",
        "keywords": ["productivity", "time management", "focus"],
        "skills": ["Productivity", "Self-Management"],
        "duration": 2400,
    },
    {
        "name": "Data Visualization with Python",
        "description": "Chart selection, perceptual best practices, and building publication-quality figures with matplotlib and plotly.",
        "keywords": ["visualization", "matplotlib", "plotly"],
        "skills": ["Data Visualization", "Python"],
        "duration": 5400,
    },
    {
        "name": "Statistics for Data Practitioners",
        "description": "Descriptive stats, distributions, hypothesis testing, and avoiding common pitfalls.",
        "keywords": ["statistics", "data analysis"],
        "skills": ["Statistics", "Data Analysis"],
        "duration": 7200,
    },
    {
        "name": "Power BI for Business Analysts",
        "description": "Data modeling, DAX measures, and building executive-ready dashboards.",
        "keywords": ["power bi", "dashboards", "bi"],
        "skills": ["Power BI", "Business Intelligence"],
        "duration": 6000,
    },
    {
        "name": "Tableau Fundamentals",
        "description": "Calculated fields, LOD expressions, dashboard layout, and storytelling with data.",
        "keywords": ["tableau", "dashboards", "bi"],
        "skills": ["Tableau", "Business Intelligence"],
        "duration": 5400,
    },
]


# ---------------------------------------------------------------------------
# Employee profiles
# ---------------------------------------------------------------------------

EMPLOYEES: list[dict] = [
    {
        "name": "Alex Müller",
        "role": "Data Engineer",
        "skills": "Python;SQL;Airflow;ETL",
        "job_description": "Build and maintain batch and streaming pipelines feeding the analytics warehouse.",
        "strengths": "Reliability mindset;Systems thinking;Mentoring",
        "interests": "Streaming systems, observability",
        "last_course": "Building Production ETL with Airflow",
    },
    {
        "name": "Priya Shah",
        "role": "Senior Software Engineer",
        "skills": "TypeScript;React;Node.js;System Design",
        "job_description": "Lead a platform team building a customer-facing web app and its backend APIs.",
        "strengths": "Architecture;Mentoring;Communication",
        "interests": "Frontend performance, API design",
        "last_course": "Modern TypeScript",
    },
    {
        "name": "Tomás Pereira",
        "role": "Analytics Engineer",
        "skills": "SQL;dbt;Python;BigQuery",
        "job_description": "Model business data in dbt and partner with analysts on metric definitions.",
        "strengths": "Attention to detail;Stakeholder communication",
        "interests": "Data quality, metric layers",
        "last_course": "Data Modeling with dbt",
    },
    {
        "name": "Maya Okafor",
        "role": "Machine Learning Engineer",
        "skills": "Python;PyTorch;MLOps;Docker",
        "job_description": "Train and deploy recommendation models for the consumer app.",
        "strengths": "Experimentation;Pragmatism",
        "interests": "LLMs, recommender systems",
        "last_course": "Practical Machine Learning with scikit-learn",
    },
    {
        "name": "Jonas Berger",
        "role": "DevOps Engineer",
        "skills": "Kubernetes;Terraform;AWS;CI/CD",
        "job_description": "Own developer-facing platform and infrastructure for the engineering org.",
        "strengths": "Reliability;Automation mindset",
        "interests": "Platform engineering, observability",
        "last_course": "Kubernetes in Production",
    },
    {
        "name": "Sara El-Hashimi",
        "role": "Product Manager",
        "skills": "Product Management;User Research;Roadmapping",
        "job_description": "Lead the discovery and delivery of new growth features.",
        "strengths": "Customer empathy;Prioritization",
        "interests": "Behavioral economics, growth",
        "last_course": "Product Management Foundations",
    },
    {
        "name": "Wei Chen",
        "role": "UX Designer",
        "skills": "Figma;UX Research;Prototyping",
        "job_description": "Design flows and components for the customer mobile app.",
        "strengths": "Visual systems;Collaboration",
        "interests": "Design systems, accessibility",
        "last_course": "Figma for Product Designers",
    },
    {
        "name": "Hannah Bauer",
        "role": "Engineering Manager",
        "skills": "Leadership;System Design;Coaching",
        "job_description": "Manage two backend teams owning checkout and payments.",
        "strengths": "1:1 coaching;Cross-team alignment",
        "interests": "Org design, distributed systems",
        "last_course": "Leading Engineering Teams",
    },
    {
        "name": "Lucas Almeida",
        "role": "Junior Data Analyst",
        "skills": "SQL;Excel;Tableau",
        "job_description": "Support marketing and ops teams with ad-hoc analyses and weekly dashboards.",
        "strengths": "Curiosity;Speed",
        "interests": "Causal inference, A/B testing",
        "last_course": "Tableau Fundamentals",
    },
    {
        "name": "Nina Petrova",
        "role": "Security Engineer",
        "skills": "AppSec;OWASP;Python;AWS",
        "job_description": "Lead application security reviews and incident response for product engineering.",
        "strengths": "Threat modeling;Cross-team collaboration",
        "interests": "Cloud security, supply-chain risk",
        "last_course": "Application Security Fundamentals",
    },
    {
        "name": "Daniel Kim",
        "role": "Backend Engineer",
        "skills": "Go;Postgres;API Design;Kafka",
        "job_description": "Build high-throughput services for the events platform.",
        "strengths": "Performance optimization;Clear writing",
        "interests": "Event-driven architecture, observability",
        "last_course": "Streaming Data with Kafka",
    },
    {
        "name": "Ines Janssen",
        "role": "Business Analyst",
        "skills": "SQL;Power BI;Stakeholder Management",
        "job_description": "Translate business questions into analyses for finance and operations.",
        "strengths": "Listening;Storytelling with data",
        "interests": "Forecasting, planning",
        "last_course": "Power BI for Business Analysts",
    },
    {
        "name": "Rafael Souza",
        "role": "Site Reliability Engineer",
        "skills": "Kubernetes;Observability;Go;Incident Response",
        "job_description": "Improve the reliability and observability of the search platform.",
        "strengths": "Calm under incidents;Root-cause focus",
        "interests": "Tracing, chaos engineering",
        "last_course": "Observability with OpenTelemetry",
    },
    {
        "name": "Sophie Laurent",
        "role": "Data Scientist",
        "skills": "Python;Statistics;Machine Learning;SQL",
        "job_description": "Build forecasting and segmentation models for the marketing team.",
        "strengths": "Statistical rigor;Communication",
        "interests": "Causal inference, Bayesian methods",
        "last_course": "Statistics for Data Practitioners",
    },
    {
        "name": "Mateusz Nowak",
        "role": "QA Engineer",
        "skills": "Testing;Selenium;Python;CI/CD",
        "job_description": "Own end-to-end and integration testing for the web platform.",
        "strengths": "Attention to detail;Process improvement",
        "interests": "Test automation, performance testing",
        "last_course": "Test-Driven Development in Python",
    },
    {
        "name": "Ayşe Yılmaz",
        "role": "Tech Lead",
        "skills": "System Design;TypeScript;Leadership;Architecture",
        "job_description": "Lead architecture and technical direction for the integrations team.",
        "strengths": "Strategic thinking;Mentoring",
        "interests": "Distributed systems, API ergonomics",
        "last_course": "System Design Interview Prep",
    },
    {
        "name": "Brian O'Connor",
        "role": "Junior Software Engineer",
        "skills": "JavaScript;React;HTML;CSS",
        "job_description": "Implement frontend features alongside senior engineers on the web team.",
        "strengths": "Eagerness to learn;Collaboration",
        "interests": "Accessibility, web performance",
        "last_course": "React with Hooks and Suspense",
    },
    {
        "name": "Linnea Karlsson",
        "role": "Cloud Architect",
        "skills": "AWS;Terraform;Security;System Design",
        "job_description": "Define cloud reference architectures and review designs across product teams.",
        "strengths": "Holistic design;Cost awareness",
        "interests": "Multi-region failover, FinOps",
        "last_course": "Terraform for Infrastructure as Code",
    },
    {
        "name": "Carlos Rivera",
        "role": "Sales Engineer",
        "skills": "Communication;Demos;SQL;Cloud",
        "job_description": "Run technical discovery and demos for enterprise prospects.",
        "strengths": "Listening;Storytelling",
        "interests": "Discovery frameworks, deal strategy",
        "last_course": "Negotiation Skills for Professionals",
    },
    {
        "name": "Yuki Tanaka",
        "role": "AI Research Engineer",
        "skills": "PyTorch;LLMs;Deep Learning;Python",
        "job_description": "Prototype LLM-powered features and evaluate model trade-offs.",
        "strengths": "Curiosity;Rapid prototyping",
        "interests": "RAG, agentic systems",
        "last_course": "LLMs and Retrieval-Augmented Generation",
    },
    {
        "name": "Marta Kovač",
        "role": "Project Manager",
        "skills": "Project Management;Agile;Stakeholder Management",
        "job_description": "Coordinate multi-team programs from kickoff to delivery.",
        "strengths": "Organization;Conflict resolution",
        "interests": "Program risk management",
        "last_course": "Agile and Scrum Essentials",
    },
    {
        "name": "Olu Adeyemi",
        "role": "Mobile Engineer",
        "skills": "Kotlin;Android;System Design",
        "job_description": "Build and ship features in the Android consumer app.",
        "strengths": "Craftsmanship;User empathy",
        "interests": "Offline-first apps, performance",
        "last_course": "Clean Code Principles",
    },
    {
        "name": "Eva Schmidt",
        "role": "Solutions Architect",
        "skills": "AWS;Kubernetes;Communication;System Design",
        "job_description": "Partner with strategic customers on platform adoption and migration.",
        "strengths": "Translation between business and tech",
        "interests": "Cloud economics, migration patterns",
        "last_course": "AWS Cloud Practitioner Essentials",
    },
    {
        "name": "Aiden Walsh",
        "role": "Working Student (Engineering)",
        "skills": "Python;Git;SQL",
        "job_description": "Support the data platform team part-time alongside university studies.",
        "strengths": "Quick learner;Reliable",
        "interests": "Backend development, ML",
        "last_course": "Time Management for Knowledge Workers",
    },
    {
        "name": "Reema Patel",
        "role": "Data Platform Lead",
        "skills": "Spark;Kafka;Python;Leadership",
        "job_description": "Lead a team building shared data infrastructure for the org.",
        "strengths": "Architecture;Mentoring;Roadmapping",
        "interests": "Lakehouse architecture, streaming",
        "last_course": "Spark for Big Data Processing",
    },
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    content_rows = []
    for i, c in enumerate(CONTENT, start=1):
        content_rows.append(
            {
                "content_id": f"C{i:04d}",
                "content_name": c["name"],
                "content_description": c["description"],
                "content_language": "en",
                "duration_seconds": c["duration"],
                "keywords": ";".join(c["keywords"]),
                "skills": ";".join(c["skills"]),
            }
        )

    employee_rows = []
    for i, e in enumerate(EMPLOYEES, start=1):
        employee_rows.append(
            {
                "employee_id": f"E{i:04d}",
                "name": e["name"],
                "role": e["role"],
                "skills": e["skills"],
                "job_description": e["job_description"],
                "strengths": e["strengths"],
                "interests": e["interests"],
                "last_course": e["last_course"],
            }
        )

    content_path = args.data_dir / "learning_content.csv"
    employees_path = args.data_dir / "employees.csv"

    write_csv(
        content_path,
        content_rows,
        fieldnames=[
            "content_id",
            "content_name",
            "content_description",
            "content_language",
            "duration_seconds",
            "keywords",
            "skills",
        ],
    )
    write_csv(
        employees_path,
        employee_rows,
        fieldnames=[
            "employee_id",
            "name",
            "role",
            "skills",
            "job_description",
            "strengths",
            "interests",
            "last_course",
        ],
    )

    print(f"Wrote {len(content_rows)} content rows to {content_path}")
    print(f"Wrote {len(employee_rows)} employee rows to {employees_path}")


if __name__ == "__main__":
    main()
