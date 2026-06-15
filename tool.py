"""
SOUP Analysis Tool
------------------
AI-driven initial SOUP listing for medical device software.
Implements — Initial Risk Profiling (IEC 62304).

Usage:
    python tool.py --description project_desc.md --sbom sbom.json
    python tool.py --description project_desc.md --sbom sbom.json --provider claude
    python tool.py --description project_desc.md --sbom sbom.json --source ./myproject
"""

import argparse
import json
import os
import re
import sys
import time
import subprocess

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pandas", "packaging", "anthropic", "google-genai"]:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import pandas as pd
from packaging.requirements import Requirement
import anthropic
from google import genai as google_genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-driven SOUP analysis tool for medical device software (IEC 62304)."
    )
    parser.add_argument(
        "--description", required=True,
        help="Path to project description file (.md or .txt)"
    )
    parser.add_argument(
        "--sbom", required=True,
        help="Path to SBOM file (CycloneDX JSON from syft) or requirements.txt"
    )
    parser.add_argument(
        "--source", default=None,
        help="Path to project source directory for reachability scan (optional)"
    )
    parser.add_argument(
        "--provider", default="gemini", choices=["gemini", "claude"],
        help="AI provider to use (default: gemini)"
    )
    parser.add_argument(
        "--output", default="soup_register.csv",
        help="Output CSV file (default: soup_register.csv)"
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# SBOM / manifest ingestion
# ---------------------------------------------------------------------------

def parse_sbom(sbom_path):
    """Read a CycloneDX SBOM. Returns list of component dicts."""
    if not os.path.isfile(sbom_path):
        print(f"[WARN] {sbom_path} not found.")
        return []
    with open(sbom_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("bomFormat") != "CycloneDX":
        print("[WARN] File does not appear to be a CycloneDX SBOM.")
        return []
    components = []
    for comp in data.get("components", []):
        name     = comp.get("name", "").strip()
        version  = comp.get("version", "").strip() or "unpinned"
        purl     = comp.get("purl", "")
        supplier = comp.get("supplier", {}).get("name", "") if comp.get("supplier") else ""
        if not supplier:
            supplier = "PyPI / " + name
        if name:
            components.append({"name": name, "version": version,
                                "purl": purl, "supplier": supplier})
    print(f"[INFO] SBOM parsed: {len(components)} components found.")
    return components


def parse_manifest(manifest_path):
    """
    Parse any manifest file the AI cannot parse directly.
    Supports: requirements.txt
    For other formats (package.json, go.mod, Cargo.toml) — raw content
    is passed to the AI to parse.
    """
    components = []
    ext = os.path.splitext(manifest_path)[1].lower()
    fname = os.path.basename(manifest_path).lower()

    if fname == "requirements.txt" or ext == ".txt":
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "-")):
                    continue
                line = line.split("#")[0].strip()
                try:
                    req     = Requirement(line)
                    specs   = list(req.specifier)
                    version = specs[0].version if specs else "unpinned"
                    components.append({"name": req.name, "version": version,
                                        "purl": "", "supplier": "PyPI / " + req.name})
                except Exception:
                    name = re.split(r"[>=<!\[\s]", line)[0].strip()
                    if name:
                        components.append({"name": name, "version": "unpinned",
                                            "purl": "", "supplier": "PyPI / " + name})
        print(f"[INFO] requirements.txt parsed: {len(components)} components.")
        return components

    # For all other manifest types — return raw content for AI to parse
    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()  # raw string — AI will handle it


def load_manifest(sbom_path):
    """
    Smart loader: tries CycloneDX SBOM first, then requirements.txt parser,
    then returns raw content for AI to handle.
    """
    ext   = os.path.splitext(sbom_path)[1].lower()
    fname = os.path.basename(sbom_path).lower()

    if ext == ".json":
        components = parse_sbom(sbom_path)
        if components:
            return components, "sbom"
        # JSON but not CycloneDX — pass raw to AI
        with open(sbom_path, "r", encoding="utf-8") as f:
            return f.read(), "raw"

    result = parse_manifest(sbom_path)
    if isinstance(result, list):
        return result, "requirements"
    return result, "raw"  # raw string for AI


# ---------------------------------------------------------------------------
# Source code reachability scan
# ---------------------------------------------------------------------------

def scan_direct_imports(project_dir):
    """
    Scan all .py files and return a dict of { package: [files] }.
    Detects directly imported packages — e.g. SciPy imported in
    feature_extractor.py even if not in requirements.txt.
    """
    import_map = {}
    import_re  = re.compile(r"^\s*import\s+([\w]+)", re.MULTILINE)
    from_re    = re.compile(r"^\s*from\s+([\w]+)", re.MULTILINE)

    if not os.path.isdir(project_dir):
        print(f"[WARN] Source directory not found: {project_dir}")
        return import_map

    py_files = 0
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and d not in ("__pycache__", ".git", "node_modules",
                                  "venv", ".venv", "build", "dist")]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    src = fh.read()
                py_files += 1
                for match in import_re.findall(src) + from_re.findall(src):
                    pkg = match.lower().split(".")[0]
                    import_map.setdefault(pkg, [])
                    rel = os.path.relpath(fpath, project_dir)
                    if rel not in import_map[pkg]:
                        import_map[pkg].append(rel)
            except Exception:
                continue

    print(f"[INFO] Source scan: {py_files} .py files, {len(import_map)} unique imports.")
    return import_map


def get_reachability(name, import_map):
    n1 = name.lower().replace("-", "_")
    n2 = name.lower().replace("-", "").replace("_", "")
    files = (import_map.get(n1) or import_map.get(name.lower()) or import_map.get(n2) or [])
    return len(files) > 0, ", ".join(files[:3]) + ("..." if len(files) > 3 else "")


# ---------------------------------------------------------------------------
# Tool / SOUP filter
# ---------------------------------------------------------------------------

TOOL_KEYWORDS = {
    "gcc", "pytest", "pylint", "black", "compiler", "tqdm", "flake8",
    "mypy", "coverage", "setuptools", "wheel", "twine", "sphinx", "isort",
    "pip", "virtualenv", "build", "mock", "faker", "hypothesis",
}

def is_tool(name):
    n = name.lower().replace("-", "").replace("_", "")
    return any(kw.replace("-", "").replace("_", "") in n for kw in TOOL_KEYWORDS)


# ---------------------------------------------------------------------------
# AI Classification
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM = """You are a Senior Medical Device Software Safety Engineer performing
initial SOUP analysis under IEC 62304.

You will be given a SOUP component from a medical device software project.
Your job is to:
1. Determine the IEC 62304 Safety Class (A, B, or C) based on the component role
   in the device given the project description.
   - Class C: failure could cause serious injury or death
   - Class B: failure could cause non-serious injury
   - Class A: no patient safety impact
2. Describe the intended use of this component in this specific project.
3. Note any patient safety concerns.

Do NOT use hardcoded knowledge of how specific projects classify components.
Determine the class from the project description and component context provided.

OUTPUT: valid JSON only, no markdown, no extra text:
{
  "class": "C",
  "intended_use": "one sentence: component role in this device",
  "rationale": "one sentence: why this class",
  "patient_safety_concern": "one sentence or None"
}"""


def classify_claude(client, model, component, version, supplier,
                     directly_imported, import_files, project_context):
    reach = (f"Directly imported: Yes — found in {import_files}"
             if directly_imported else "Directly imported: No — transitive dependency")
    prompt = (
        f"PROJECT DESCRIPTION:\n{project_context}\n\n"
        f"COMPONENT:\n"
        f"  Name: {component} v{version}\n"
        f"  Supplier: {supplier}\n"
        f"  {reach}\n\n"
        f"Assign the IEC 62304 safety class."
    )
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model, max_tokens=512,
                system=CLASSIFICATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip().replace("```json","").replace("```","").strip()
            return json.loads(text)
        except Exception as e:
            if attempt == 2:
                return {"class": "B", "intended_use": "Unknown",
                        "rationale": str(e), "patient_safety_concern": "Evaluation failed"}
            time.sleep(2)


def classify_gemini(client, model, component, version, supplier,
                     directly_imported, import_files, project_context):
    reach = (f"Directly imported: Yes — found in {import_files}"
             if directly_imported else "Directly imported: No — transitive dependency")
    prompt = (
        f"PROJECT DESCRIPTION:\n{project_context}\n\n"
        f"COMPONENT:\n"
        f"  Name: {component} v{version}\n"
        f"  Supplier: {supplier}\n"
        f"  {reach}\n\n"
        f"Assign the IEC 62304 safety class."
    )
    full_prompt = CLASSIFICATION_SYSTEM + "\n\n" + prompt
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model, contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=512, temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            text = response.text.strip().replace("```json","").replace("```","").strip()
            return json.loads(text)
        except Exception as e:
            if attempt == 2:
                return {"class": "B", "intended_use": "Unknown",
                        "rationale": str(e), "patient_safety_concern": "Evaluation failed"}
            time.sleep(2)


# ---------------------------------------------------------------------------
# Agentic chat loop
# ---------------------------------------------------------------------------

def run_chat_agent(df_soup, project_description, provider,
                    claude_client, gemini_client,
                    claude_model, gemini_model, output_file):
    """
    Terminal chat agent — user can ask questions about the SOUP register,
    request reclassification, filter components, and export results.
    The agent maintains conversation history for context.
    """
    conversation_history = []

    # Build initial context message
    soup_summary = df_soup[["name","version","supplier","ai_class",
                              "ai_intended_use","directly_imported",
                              "compliance_gap"]].to_string(index=False)

    system_prompt = f"""You are a Senior Medical Device Software Safety Engineer.
You have completed an initial SOUP analysis for a medical device project.

PROJECT DESCRIPTION:
{project_description}

INITIAL SOUP LISTING (already classified):
{soup_summary}

You can answer questions about the SOUP components, explain classifications,
suggest additional analysis, identify compliance gaps, or update classifications
if the user provides new information.

When the user asks to export or save results, confirm what will be saved.
When the user asks to reclassify a component, explain your reasoning clearly."""

    print("\n" + "="*60)
    print("SOUP Analysis Agent — Chat Mode")
    print(f"Provider: {provider.upper()}")
    print("="*60)
    print("SOUP analysis complete. Ask me anything about the results.")
    print("Examples:")
    print("  'Show me only Class C components'")
    print("  'Why is numpy classified as Class C?'")
    print("  'What are the compliance gaps?'")
    print("  'Export results'")
    print("  'exit' to quit")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "done", "bye"):
            print("\nAgent: Goodbye! Results saved to:", os.path.abspath(output_file))
            break

        if user_input.lower() in ("export", "save", "export results"):
            df_soup.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\nAgent: Saved to {os.path.abspath(output_file)}\n")
            continue

        conversation_history.append({"role": "user", "content": user_input})

        if provider == "claude":
            try:
                response = claude_client.messages.create(
                    model=claude_model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=conversation_history
                )
                reply = response.content[0].text.strip()
            except Exception as e:
                reply = f"[Error: {e}]"
        else:
            try:
                # Build full conversation for Gemini
                full = system_prompt + "\n\n"
                for msg in conversation_history:
                    role = "User" if msg["role"] == "user" else "Agent"
                    full += f"{role}: {msg['content']}\n\n"
                full += "Agent:"
                response = gemini_client.models.generate_content(
                    model=gemini_model, contents=full,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=4096, temperature=0.1)
                )
                reply = response.text.strip()
            except Exception as e:
                reply = f"[Error: {e}]"

        conversation_history.append({"role": "assistant", "content": reply})
        print(f"\nAgent: {reply}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Read API keys
    if args.provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY") or input("Anthropic API key: ").strip()
        claude_client = anthropic.Anthropic(api_key=api_key)
        gemini_client = None
        claude_model  = "claude-sonnet-4-6"
        gemini_model  = None
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or input("Gemini API key: ").strip()
        gemini_client = google_genai.Client(api_key=api_key)
        claude_client = None
        claude_model  = None
        gemini_model  = "gemini-2.5-flash"

    # Read project description
    print(f"\n[INFO] Reading project description: {args.description}")
    if not os.path.isfile(args.description):
        print(f"[ERROR] File not found: {args.description}")
        sys.exit(1)
    with open(args.description, "r", encoding="utf-8") as f:
        project_description = f.read()

    # Load SBOM or manifest
    print(f"[INFO] Loading manifest/SBOM: {args.sbom}")
    manifest_data, manifest_type = load_manifest(args.sbom)

    # Build component list
    if manifest_type in ("sbom", "requirements"):
        sbom_components = manifest_data
    else:
        # Raw manifest — AI will handle it in the chat
        sbom_components = []
        print("[INFO] Non-standard manifest — will pass raw content to AI.")

    # Source code scan
    import_map = {}
    if args.source:
        print(f"[INFO] Scanning source directory: {args.source}")
        import_map = scan_direct_imports(args.source)

    # Build SOUP register
    rows = []
    for comp in sbom_components:
        name    = comp["name"]
        version = comp["version"]
        pinned  = version not in ("", "unpinned", None)

        if is_tool(name):
            continue  # skip dev tools

        di, imf = get_reachability(name, import_map)
        gap = "VERSION UNPINNED - IEC 62304 S8.1.2 not satisfied" if not pinned else ""
        rows.append({
            "name":             name,
            "version":          version,
            "supplier":         comp["supplier"],
            "pinned":           pinned,
            "directly_imported": di,
            "import_files":     imf,
            "compliance_gap":   gap,
        })

    df_soup = pd.DataFrame(rows)

    if df_soup.empty:
        print("[WARN] No SOUP components found. Check SBOM or manifest file.")
        sys.exit(1)

    print(f"\n[INFO] SOUP components: {len(df_soup)}")
    print(f"[INFO] Directly imported: {df_soup['directly_imported'].sum()}")
    print(f"[INFO] Unpinned: {(~df_soup['pinned']).sum()}")

    # AI Classification
    print(f"\n[INFO] Classifying with {args.provider.upper()}...\n")
    project_context = (
        f"{project_description}\n\n"
        f"Components found: {', '.join(df_soup['name'].tolist())}"
    )

    ai_classes, ai_uses, ai_rationales = [], [], []

    for _, row in df_soup.iterrows():
        if args.provider == "claude":
            result = classify_claude(
                claude_client, claude_model,
                row["name"], row["version"], row["supplier"],
                row["directly_imported"], row["import_files"],
                project_context
            )
        else:
            result = classify_gemini(
                gemini_client, gemini_model,
                row["name"], row["version"], row["supplier"],
                row["directly_imported"], row["import_files"],
                project_context
            )

        ai_classes.append(result.get("class", "B"))
        ai_uses.append(result.get("intended_use", ""))
        ai_rationales.append(result.get("rationale", ""))
        print(f"  {row['name']}: Class {result.get('class','?')} — {result.get('intended_use','')[:60]}")
        time.sleep(0.3)

    df_soup["ai_class"]        = ai_classes
    df_soup["ai_intended_use"] = ai_uses
    df_soup["ai_rationale"]    = ai_rationales

    # Save initial output
    df_soup.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Initial SOUP register saved: {os.path.abspath(args.output)}")

    # Print Clause 7.1.3 table
    print("\n" + "="*60)
    print("IEC 62304 Clause 7.1.3 — Initial SOUP Listing")
    print("="*60)
    for _, row in df_soup.iterrows():
        print(f"\n  {row['name']} v{row['version']}")
        print(f"  Supplier:      {row['supplier']}")
        print(f"  Safety Class:  Class {row['ai_class']}")
        print(f"  Intended Use:  {row['ai_intended_use']}")
        print(f"  Direct Import: {'Yes' if row['directly_imported'] else 'No'}")
        if row["compliance_gap"]:
            print(f"  [GAP] {row['compliance_gap']}")

    # Start chat agent
    run_chat_agent(
        df_soup=df_soup,
        project_description=project_description,
        provider=args.provider,
        claude_client=claude_client,
        gemini_client=gemini_client,
        claude_model=claude_model,
        gemini_model=gemini_model,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
