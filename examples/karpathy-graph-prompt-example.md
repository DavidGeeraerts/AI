# RESEARCH NOTE · KARPATHY-INSPIRED GRAPH PROMPT · OPUS 5.5

**OpenAI Founding Member**  
**Andrej Karpathy** · Claude Code & agent graphs

---

```xml
<prompt>
<goal>
Complete {{TASK}} as a graph of bounded nodes, not one long session.
Leave a usable result, a verification report, and graph.md so another
session can rerun or extend it. Define completion before starting.
Stop when every node’s checks pass or the agreed limit is reached.
</goal>

<role>
Act as the lead that designs, runs, and maintains this graph. Read the
existing project instructions and inspect current work before editing.
Plan and merge yourself; give bounded work to subagents. Record
decisions that change the graph.
</role>

<principles>
Adapt these ideas from Karpathy’s autoresearch:
- Establish a baseline before changing the system.
- Give every node one clear scope and a fixed evaluator.
- Change one thing, measure it, and keep useful improvements. Record
  failed attempts too.
- Prefer fewer nodes when results are comparable.
The graph setup below is my adaptation for multi-agent work in Claude
Code. [1, 2]
</principles>

<inputs>
Task: {{TASK}}
Deliverable and output path: {{OUTPUT}}
Acceptance checks: {{CHECKS}}
Source material: {{SOURCES}}
Allowed changes: {{SCOPE}}
Run limit: {{LIMIT}}
Fill these fields before starting. If a missing detail changes the
outcome, ask one focused question.
</inputs>

<layer_1_split>
Break {{TASK}} into nodes before any work starts. A node is one
bounded job: one input, one output, one check.
- Split by what each node must read, not by folder.
- Draw an edge only when a node consumes another node’s output.
- Mark nodes with no edge between them as parallel.
Write nodes, edges, and checks into graph.md and pause for approval.
</layer_1_split>

<layer_2_nodes>
Run each node as its own subagent with only the context it needs:
1. Pass the node spec, its inputs, and its check.
2. Build the artifact for that node only.
3. Run the node’s check against the saved artifact.
4. Return the output path, verdict, and evidence.
Keep node specs reusable across new source sets.
</layer_2_nodes>

<layer_3_code_nodes>
If a step has exactly one correct answer (merge, sort, dedupe,
compare, count), write it as code, not as a model call. A model call
is for judgment only.
- List every code node in graph.md.
- Never route a deterministic step through a subagent.
If a required tool is unavailable, name the blocker and preserve the
nodes that can still be completed.
</layer_3_code_nodes>

<layer_4_permissions>
Check the active permissions before running the graph. Keep edits
inside {{SCOPE}}. Configure tool rules or sandbox controls when
enforcement is required; prose alone is not a boundary.
- Ask before publishing, sending, or spending.
- Never let two nodes edit the same file.
- Preserve a recoverable version before risky merges.
</layer_4_permissions>

<layer_5_gates>
Put a gate on every edge that carries risky work. Open it on evidence:
deterministic checks first, then the node’s report, the model’s
confidence last.
Save checks.md with a row for each requirement:
node | requirement | verdict | evidence
Use pass, fail, or unresolved. Include file paths or actual check
output. Leave missing evidence visible.
</layer_5_gates>

<layer_6_return_paths>
Correction edge: a failed gate sends the unit back to the node that
made it, with the reason, the evidence, and the scope “fix this node
only”. Cap it at three attempts.
Learning edge: after an accepted result, write the rule it proves into
graph.md so the next run starts from it.
</layer_6_return_paths>

<layer_7_completion>
Before execution, write a completion checklist from {{CHECKS}}. Each
item must point to observable evidence.
- Every node’s output exists and can be opened.
- Every gate ran against the saved version.
- Failures are fixed or explicitly reported.
- graph.md matches the nodes that actually ran.
If a limit or blocker stops the run, return a partial status with the
exact nodes left.
</layer_7_completion>

<execution_loop>
1. Read graph.md and pick the next ready nodes.
2. Run independent nodes in parallel.
3. Pass every output through its gate.
4. Return failed units, never the whole batch.
5. Merge accepted outputs in code.
6. Continue until completion, a blocker, or {{LIMIT}}.
Do not weaken a gate to obtain a pass.
</execution_loop>

<handoff>
After each stage, update progress.md:
Graph: nodes done, running, and blocked.
Outputs: exact paths to the current saved files.
Decisions: what changed in graph.md and why.
Open issues: failures, uncertainties, and blockers.
Next action: the next ready node.
On resumption, read graph.md and progress.md first, then continue from
the recorded next node.
</handoff>

<delivery>
Return {{OUTPUT}}, graph.md, checks.md, and progress.md. State what
was checked, what passed, and what remains unresolved. Save the graph
as a reusable template after review. Report measured usage only when
available; do not infer savings from the prompt.
</delivery>
</prompt>