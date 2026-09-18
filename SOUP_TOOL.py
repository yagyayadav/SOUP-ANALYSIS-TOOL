"""
SOUP Analysis Tool 
--------------------------------------------------------
Automated initial SOUP listing, version audit, and runtime I/O safety profiling 
for medical device software under IEC 62304 (§5.3.3 / §8.1.2) and ISO 14971.
Usage:
    python SOUP_TOOL.py --description project_desc.md --sbom sbom.json --source ./mimic3-benchmarks --provider gemini
    python SOUP_TOOL.py --description project_desc.md --sbom sbom.json --source ./mimic3-benchmarks --provider claude
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Install required packages if not present
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pandas", "anthropic", "google-genai"]:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import pandas as pd
import anthropic
from google import genai as google_genai
from google.genai import types as genai_types


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-driven SOUP and dynamic I/O compliance analysis tool (IEC 62304 & ISO 14971)."
    )
    parser.add_argument("--description", required=True,
        help="Path to project description file (.md or .txt)")
    parser.add_argument("--sbom", required=True,
        help="Path to SBOM or manifest file — any format, AI will parse it")
    parser.add_argument("--source", default=".",
        help="Project source directory for grep commands (default: current dir)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude"],
        help="AI provider (default: gemini)")
    parser.add_argument("--output", default="soup_compliance_register.csv",
        help="Output CSV filename (default: soup_compliance_register.csv)")
    return parser.parse_args()


def read_file(path):
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def run_subcommand(command, source_dir):
    # Replace any placeholder paths with the actual source directory
    command = command.replace("/path/to/project", source_dir)
    command = command.replace("<project_dir>", source_dir)
    command = command.replace("$PROJECT", source_dir)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"[Error running command: {e}]"


def call_claude(client, model, system, messages, max_tokens=8192):
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model, max_tokens=max_tokens,
                system=system, messages=messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            if attempt == 2:
                return f"[Error: {e}]"
            time.sleep(2)


def call_gemini(client, model, prompt, max_tokens=8192, json_mode=False):
    for attempt in range(3):
        try:
            config = genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
                response_mime_type="application/json" if json_mode else "text/plain"
            )
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            return response.text.strip()
        except Exception as e:
            if attempt == 2:
                return f"[Error: {e}]"
            time.sleep(2)



# Step 1 : Initial classification prompt
CLASSIFICATION_SYSTEM = """You are a Medical Device Software Safety Engineer performing
an automated initial SOUP and compliance audit under IEC 62304 and ISO 14971.

You will be given a project description and a manifest/SBOM file in any format.

Your tasks:
1. Parse the manifest dynamically and identify all third-party dependencies.
2. Filter Noise: Separate genuine runtime SOUP from development, testing, or compiler tools (e.g., pytest, black, gcc, pylint are NOT runtime SOUP).
3. Base Safety Classification (IEC 62304):
   - Class C: Failure can directly contribute to serious patient injury or death (e.g., core ML calculations, array engines, execution interpreters).
   - Class B: Failure can contribute to non-serious patient injury.
   - Class A: Failure cannot contribute to injury (e.g., loading bars/tqdm, static logging config).
4. Identify IEC 62304 §8.1.2 Compliance Gaps: If a component version is 'unpinned' or 'unspecified', flag it as "VERSION UNPINNED - IEC 62304 §8.1.2 not satisfied".
5. Focus strictly on Input/Output Patient Safety Impact: Skip traditional risk probability calculations. Focus on how incorrect mathematical calculations, formatting errors, shape/dimensional mismatches, or silent NaN propagation in this SOUP's output can corrupt the downstream model's clinical predictions.
6. Generate Code-level Gating Strategies (Mitigations): Write concrete programming logic (assertions, type checks, or value-range gating) to physically isolate and validate this SOUP's outputs before they hit the core medical device code.

OUTPUT format: Respond ONLY with valid JSON inside a single JSON block. Do not write markdown prose around it.
{
  "components": [
    {
      "name": "numpy",
      "version": "1.16.5",
      "supplier": "NumPy developers",
      "class": "C",
      "intended_use": "Core numerical array processing across features, training, and inference pipelines.",
      "compliance_gap": "",
      "io_safety_impact": "Floating-point precision errors or mathematical shape mismatches in matrices pass corrupted numerical states to the downstream clinical risk prediction model, leading to wrong diagnostic evaluations.",
      "required_io_controls": "assert isinstance(arr, np.ndarray); assert not np.isnan(arr).any(); assert arr.shape == expected_shape"
    }
  ]
}
"""


def initial_classification(provider, claude_client, gemini_client,
                            claude_model, gemini_model,
                            project_description, manifest_content):
    prompt_user = (
        f"PROJECT DESCRIPTION:\n{project_description}\n\n"
        f"MANIFEST/SBOM FILE CONTENT:\n{manifest_content}\n\n"
        f"Parse this manifest, identify all SOUP components, audit their versions, evaluate their I/O safety impacts, and generate runtime control strategies."
    )
    if provider == "claude":
        text = call_claude(claude_client, claude_model, CLASSIFICATION_SYSTEM,
                           [{"role": "user", "content": prompt_user}])
    else:
        text = call_gemini(gemini_client, gemini_model,
                           CLASSIFICATION_SYSTEM + "\n\n" + prompt_user,
                           json_mode=True)
    try:
        print("\n--- Raw AI Response Output ---")
        print(text)
        print("------------------------------\n")
        
        # Robust stripping of markdown ticks to prevent JSON loads failure
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        return json.loads(cleaned_text).get("components", [])
    except Exception as e:
        print(f"[JSON Parse Error / Recovery Attempted]: {e}")
        return []


# Step 2 — Interrogative Source Code Context Gathering loop
CONTEXT_SYSTEM = """You are a Medical Device Software Safety Engineer.
You have completed an initial SOUP classification but need active source code context
to avoid hallucination — you need to know exactly which files are directly importing or using each library.

You can suggest grep commands that the user will run on their machine to expand your context.
The user will approve and run it. You will receive the real output.
Please provide suggested commands in the following format, and only suggest one command at a time:
SUGGEST_COMMAND: <command here>

Example commands could be:
SUGGEST_COMMAND: grep -rn "import numpy" <project_dir> --include="*.py"
SUGGEST_COMMAND: grep -rn "import scipy" <project_dir> --include="*.py"

Your goal is to find where and how each SOUP component is used in the source code to verify your classifications.
When you have enough context for all components (specifically validating transitive dependencies like SciPy that might be directly imported), say exactly:
CONTEXT_COMPLETE

Do not classify based on assumptions — only on evidence from project description and actual search results."""


def context_gathering_loop(provider, claude_client, gemini_client,
                            claude_model, gemini_model,
                            project_description, components, source_dir):
    print("\n" + "="*60)
    print("Step 2 — Source Code Context Gathering (Interrogative Loop)")
    print("="*60)
    print("Agent will suggest search commands to verify active code usage.")
    print("You approve each command before it executes.")
    print("Type 'skip' to go straight to interactive chat.")
    print("="*60 + "\n")

    comp_summary = json.dumps(components, indent=2)
    conversation = []

    initial_msg = (
        f"PROJECT DESCRIPTION:\n{project_description}\n\n"
        f"SOURCE DIRECTORY: {source_dir}\n\n"
        f"INITIAL SOUP CLASSIFICATION:\n{comp_summary}\n\n"
        f"Review the classifications. Suggest grep commands to verify "
        f"where safety-critical components (Class C first, or suspected direct imports like SciPy) are used. "
        f"Use {source_dir} as the project path in your commands."
    )
    conversation.append({"role": "user", "content": initial_msg})

    max_rounds = 10
    for _ in range(max_rounds):
        if provider == "claude":
            reply = call_claude(claude_client, claude_model,
                                CONTEXT_SYSTEM, conversation, max_tokens=1024)
        else:
            full = CONTEXT_SYSTEM + "\n\n"
            for msg in conversation:
                role = "User" if msg["role"] == "user" else "Agent"
                full += f"{role}: {msg['content']}\n\n"
            full += "Agent:"
            reply = call_gemini(gemini_client, gemini_model, full, max_tokens=1024)

        print(f"Agent: {reply}\n")
        conversation.append({"role": "assistant", "content": reply})

        if "CONTEXT_COMPLETE" in reply:
            print("[INFO] Agent has enough context. Moving to chat...\n")
            break

        if "SUGGEST_COMMAND:" in reply:
            lines = reply.split("\n")
            cmd_line = next((l for l in lines if "SUGGEST_COMMAND:" in l), None)
            if cmd_line:
                suggested_cmd = cmd_line.replace("SUGGEST_COMMAND:", "").strip()
                print(f"[COMMAND]: {suggested_cmd}")
                user_input = input("Run? (yes / no / skip): ").strip().lower()

                if user_input == "skip":
                    print("[INFO] Skipping. Moving to chat...\n")
                    break

                if user_input in ("yes", "y"):
                    print("[Running...]\n")
                    cmd_output = run_subcommand(suggested_cmd, source_dir)
                    print(f"[Output]:\n{cmd_output}\n")
                    feedback = (
                        f"Command output:\n{cmd_output}\n\n"
                        f"Update your analysis based on this. "
                        f"Suggest another command if needed or say CONTEXT_COMPLETE."
                    )
                    conversation.append({"role": "user", "content": feedback})
                else:
                    conversation.append({
                        "role": "user",
                        "content": "Command skipped. Suggest another or say CONTEXT_COMPLETE."
                    })
        else:
            user_input = input("You (Enter to continue / 'skip' for chat): ").strip()
            if user_input.lower() == "skip":
                break
            if user_input:
                conversation.append({"role": "user", "content": user_input})
            else:
                conversation.append({
                    "role": "user",
                    "content": "Please suggest a command or say CONTEXT_COMPLETE."
                })

    return conversation


# Step 3 — Chat Agent starts after context is gathered
CHAT_SYSTEM_TEMPLATE = """You are a Medical Device Software Safety Engineer.
You have completed a structured SOUP I/O and compliance audit.

PROJECT DESCRIPTION:
{project_description}

SOUP LISTING & COMPLIANCE REGISTER:
{soup_summary}

Answer questions about SOUP safety classifications, explain unpinned §8.1.2 gaps,
discuss why certain packages are down-classified as Non-SOUP (Class A),
or generate technical text for regulatory submissions.

If you need more source code context, you can still run:
SUGGEST_COMMAND: grep -rn "import library" {source_dir} --include="*.py"
"""


def run_chat_agent(df_soup, project_description, context_conversation,
                    provider, claude_client, gemini_client,
                    claude_model, gemini_model, output_file, source_dir):

    soup_summary = df_soup.to_string(index=False)
    system_prompt = CHAT_SYSTEM_TEMPLATE.format(
        project_description=project_description,
        soup_summary=soup_summary,
        source_dir=source_dir
    )

    conversation_history = context_conversation.copy()

    print("\n" + "="*60)
    print(f"Step 3 — Interactive Compliance Chat — {provider.upper()}")
    print("="*60)
    print("Ask about classifications, I/O safety bounds, or draft reports.")
    print("'export' — save CSV  |  'exit' — quit")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "done"):
            df_soup.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\nAgent: Register saved to {os.path.abspath(output_file)}")
            break

        if user_input.lower() in ("export", "save"):
            df_soup.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\nAgent: Register saved to {os.path.abspath(output_file)}\n")
            continue

        conversation_history.append({"role": "user", "content": user_input})

        if provider == "claude":
            reply = call_claude(claude_client, claude_model,
                                system_prompt, conversation_history, max_tokens=2048)
        else:
            full = system_prompt + "\n\n"
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Agent"
                full += f"{role}: {msg['content']}\n\n"
            full += "Agent:"
            reply = call_gemini(gemini_client, gemini_model, full, max_tokens=2048)

        if "SUGGEST_COMMAND:" in reply:
            lines = reply.split("\n")
            cmd_line = next((l for l in lines if "SUGGEST_COMMAND:" in l), None)
            if cmd_line:
                suggested_cmd = cmd_line.replace("SUGGEST_COMMAND:", "").strip()
                print(f"\nAgent: {reply}\n")
                print(f"[COMMAND]: {suggested_cmd}")
                approval = input("Run? (yes/no): ").strip().lower()
                if approval in ("yes", "y"):
                    print("[Running...]\n")
                    cmd_output = run_subcommand(suggested_cmd, source_dir)
                    print(f"[Output]:\n{cmd_output}\n")
                    conversation_history.append({"role": "assistant", "content": reply})
                    conversation_history.append({
                        "role": "user",
                        "content": f"Command output:\n{cmd_output}\nPlease update analysis."
                    })
                    continue
                else:
                    print("[Skipped]\n")
        else:
            print(f"\nAgent: {reply}\n")

        conversation_history.append({"role": "assistant", "content": reply})


def main():
    args = parse_args()

    # API authentication and Setup
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
        gemini_model  = "gemini-2.5-flash" # Use latest stable Gemini 2.5 engine

    print(f"\n[INFO] Reading project description: {args.description}")
    project_description = read_file(args.description)

    print(f"[INFO] Reading manifest/SBOM: {args.sbom}")
    manifest_content = read_file(args.sbom)

    print(f"[INFO] Source directory: {args.source}")
    print(f"[INFO] Provider: {args.provider.upper()}")

    # Step 1 — Parsing, filtering, and multi-layered safety profile audit
    print(f"\n[Step 1] Ingesting manifest dynamically and executing safety audits...\n")
    components = initial_classification(
        args.provider, claude_client, gemini_client,
        claude_model, gemini_model,
        project_description, manifest_content
    )

    if not components:
        print("[ERROR] No components could be analyzed. Please check your configurations and API Keys.")
        sys.exit(1)

    df_soup = pd.DataFrame(components)
    print(f"[INFO] {len(df_soup)} SOUP/dependencies parsed.")

    print("\n" + "="*60)
    print("IEC 62304 Compliance Register (Initial Multi-Layered Risk Profiling)")
    print("="*60)
    for _, row in df_soup.iterrows():
        print(f"\n  {row.get('name','')} v{row.get('version','')}")
        print(f"  Supplier:         {row.get('supplier','')}")
        print(f"  Safety Class:     {row.get('class','')}")
        print(f"  Intended Use:     {row.get('intended_use','')}")
        print(f"  I/O Safety Impact: {row.get('io_safety_impact','')}")
        print(f"  Gating Mitigations: {row.get('required_io_controls','')}")
        if row.get("compliance_gap"):
            print(f"  [COMPLIANCE GAP] {row.get('compliance_gap','')}")

    df_soup.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Compliance Register saved to: {os.path.abspath(args.output)}")

    # Step 2 — source code context gathering before Chat
    context_conversation = context_gathering_loop(
        args.provider, claude_client, gemini_client,
        claude_model, gemini_model,
        project_description, components, args.source
    )

    # Step 3 — Interactive Chat Agent
    run_chat_agent(
        df_soup=df_soup,
        project_description=project_description,
        context_conversation=context_conversation,
        provider=args.provider,
        claude_client=claude_client,
        gemini_client=gemini_client,
        claude_model=claude_model,
        gemini_model=gemini_model,
        output_file=args.output,
        source_dir=args.source
    )


if __name__ == "__main__":
    main()