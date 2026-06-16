"""
SOUP Analysis Tool
------------------
AI-driven initial SOUP listing for medical device software.
Implements Initial Risk Profiling (IEC 62304).

Usage:
    python tool.py --description project_desc.md --sbom sbom.json
    python tool.py --description project_desc.md --sbom sbom.json --provider claude
    python tool.py --description project_desc.md --sbom sbom.json --source ./myproject
"""

import argparse
import json
import os
import subprocess
import sys
import time

# install required packages if not present
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["pandas", "anthropic", "google-genai"]:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import pandas as pd

SOURCE_DIR = "."
import anthropic
from google import genai as google_genai
from google.genai import types as genai_types


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-driven SOUP analysis tool for medical device software (IEC 62304)."
    )
    parser.add_argument("--description", required=True,
        help="Path to project description file (.md or .txt)")
    parser.add_argument("--sbom", required=True,
        help="Path to SBOM or manifest file — any format, AI will parse it")
    parser.add_argument("--source", default=".",
        help="Project source directory for grep commands (default: current dir)")
    parser.add_argument("--source", default=".",
        help="Project source directory for context commands (default: current dir)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude"],
        help="AI provider (default: gemini)")
    parser.add_argument("--output", default="soup_register.csv",
        help="Output CSV filename (default: soup_register.csv)")
    return parser.parse_args()


def read_file(path):
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def run_subcommand(command, source_dir):
    # replace any placeholder paths with the actual source directory
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


def call_claude(client, model, system, messages, max_tokens=4096):
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


def call_gemini(client, model, prompt, max_tokens=4096, json_mode=False):
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


# step 1 — AI parses manifest and does initial classification
# AI handles any format — requirements.txt, package.json, go.mod, CycloneDX JSON etc.
CLASSIFICATION_SYSTEM = """You are a Medical Device Software Safety Engineer performing
initial SOUP analysis under IEC 62304.

You will be given a project description and a manifest or SBOM file in any format
(requirements.txt, package.json, go.mod, CycloneDX JSON etc.)

Your tasks:
1. Parse the manifest and identify all third-party dependencies
2. Separate SOUP from development tools (pytest, tqdm, pylint, black etc. are NOT SOUP)
3. For each SOUP component assign IEC 62304 Safety Class:
   - Class C: failure could cause serious injury or death
   - Class B: failure could cause non-serious injury
   - Class A: no patient safety impact
4. Describe the intended use of each component in this specific project
5. Flag unpinned versions as compliance gaps under IEC 62304 S8.1.2

Base classification on the project description and component context only.

OUTPUT: valid JSON only, no markdown:
{
  "components": [
    {
      "name": "numpy",
      "version": "1.16.5",
      "supplier": "NumPy developers",
      "class": "C",
      "intended_use": "one sentence describing role in this device",
      "rationale": "one sentence explaining the class",
      "compliance_gap": ""
    }
  ]
}

Set compliance_gap to "VERSION UNPINNED - IEC 62304 S8.1.2 not satisfied" for unpinned versions.
Leave compliance_gap as empty string if version is pinned."""


def initial_classification(provider, claude_client, gemini_client,
                            claude_model, gemini_model,
                            project_description, manifest_content):
    prompt_user = (
        f"PROJECT DESCRIPTION:\n{project_description}\n\n"
        f"MANIFEST/SBOM FILE CONTENT:\n{manifest_content}\n\n"
        f"Parse this manifest, identify all SOUP components, and classify each one."
    )
    if provider == "claude":
        text = call_claude(claude_client, claude_model, CLASSIFICATION_SYSTEM,
                           [{"role": "user", "content": prompt_user}])
    else:
        text = call_gemini(gemini_client, gemini_model,
                           CLASSIFICATION_SYSTEM + "\n\n" + prompt_user,
                           json_mode=True)
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text).get("components", [])
    except Exception:
        return []


# step 2 — context gathering loop before chat starts
# AI suggests grep commands, user approves, results feed back to AI
# runs until AI says CONTEXT_COMPLETE or user skips
CONTEXT_SYSTEM = """You are a Medical Device Software Safety Engineer.
You have done an initial SOUP classification but need source code context
to avoid hallucination — you need to know where each library is actually used.

To search the source code, suggest a command using this exact format on its own line:
SUGGEST_COMMAND: grep -rn "import numpy" <project_dir> --include="*.py"

The user will approve and run it. You will receive the real output.
Use that output to update your understanding of how the library is used.

When you have enough context for all components, say exactly:
CONTEXT_COMPLETE

Do not classify based on assumptions — only on evidence from project description
and actual source code search results."""


def context_gathering_loop(provider, claude_client, gemini_client,
                            claude_model, gemini_model,
                            project_description, components, source_dir):
    print("\n" + "="*60)
    print("Step 2 — Source Code Context Gathering")
    print("="*60)
    print("Agent will suggest commands to search source code.")
    print("You approve each command before it runs.")
    print("Type 'skip' to go straight to chat.")
    print("="*60 + "\n")

    comp_summary = json.dumps(components, indent=2)
    conversation = []

    initial_msg = (
        f"PROJECT DESCRIPTION:\n{project_description}\n\n"
        f"SOURCE DIRECTORY: {source_dir}\n\n"
        f"INITIAL SOUP CLASSIFICATION:\n{comp_summary}\n\n"
        f"Review the classifications. Suggest grep commands to verify "
        f"where safety-critical components (Class C first) are actually used. "
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


# step 3 — chat agent starts after context is gathered
CHAT_SYSTEM_TEMPLATE = """You are a Medical Device Software Safety Engineer.
You have completed an initial SOUP analysis with source code context.

PROJECT DESCRIPTION:
{project_description}

SOUP LISTING:
{soup_summary}

Answer questions about SOUP components, explain classifications,
identify compliance gaps, or generate compliance report text.
If you need more source code context, use:
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

    # seed with context conversation so AI remembers what it found
    conversation_history = context_conversation.copy()

    print("\n" + "="*60)
    print(f"Step 3 — SOUP Analysis Chat — {provider.upper()}")
    print("="*60)
    print("Ask questions about the SOUP analysis.")
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
            print(f"\nAgent: Saved to {os.path.abspath(output_file)}")
            break

        if user_input.lower() in ("export", "save"):
            df_soup.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\nAgent: Saved to {os.path.abspath(output_file)}\n")
            continue

        conversation_history.append({"role": "user", "content": user_input})

        if provider == "claude":
            reply = call_claude(claude_client, claude_model,
                                system_prompt, conversation_history, max_tokens=1024)
        else:
            full = system_prompt + "\n\n"
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Agent"
                full += f"{role}: {msg['content']}\n\n"
            full += "Agent:"
            reply = call_gemini(gemini_client, gemini_model, full, max_tokens=1024)

        # handle subcommand suggestions in chat too
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

    # load api key from environment or prompt
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

    print(f"\n[INFO] Reading project description: {args.description}")
    project_description = read_file(args.description)

    print(f"[INFO] Reading manifest/SBOM: {args.sbom}")
    manifest_content = read_file(args.sbom)

    print(f"[INFO] Source directory: {args.source}")
    print(f"[INFO] Provider: {args.provider.upper()}")

    # step 1 — initial classification
    print(f"\n[Step 1] Parsing manifest and classifying SOUP components...\n")
    components = initial_classification(
        args.provider, claude_client, gemini_client,
        claude_model, gemini_model,
        project_description, manifest_content
    )

    if not components:
        print("[ERROR] No components found. Check manifest file and API key.")
        sys.exit(1)

    df_soup = pd.DataFrame(components)
    print(f"[INFO] {len(df_soup)} SOUP components identified.")

    print("\n" + "="*60)
    print("IEC 62304 Clause 7.1.3 — Initial SOUP Listing")
    print("="*60)
    for _, row in df_soup.iterrows():
        print(f"\n  {row.get('name','')} v{row.get('version','')}")
        print(f"  Supplier:     {row.get('supplier','')}")
        print(f"  Class:        {row.get('class','')}")
        print(f"  Intended Use: {row.get('intended_use','')}")
        if row.get("compliance_gap"):
            print(f"  [GAP] {row.get('compliance_gap','')}")

    df_soup.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Register saved: {os.path.abspath(args.output)}")

    # step 2 — source code context gathering before chat
    context_conversation = context_gathering_loop(
        args.provider, claude_client, gemini_client,
        claude_model, gemini_model,
        project_description, components, args.source
    )

    # step 3 — chat agent
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
