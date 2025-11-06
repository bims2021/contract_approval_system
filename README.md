# 🤖 Contract Approval System MVP: LangGraph Multi-Agent Workflow

##  Project Overview

This repository contains an **Minimum Viable Product (MVP)** for an automated **Contract Approval System** built using **LangGraph** for orchestration and a **Model Context Protocol (MCP)** server for state management and coordination.

The system processes uploaded PDF contracts, analyzes them for risk and key clauses, and routes them for either **auto-approval** (low risk) or **manual approval** (medium/high risk), ensuring all approved contracts are digitally signed.

-----

##  Features

  * **LangGraph Orchestration:** Defines a robust, multi-step workflow using LangGraph for agent sequencing and conditional routing.
  * **Three Core Agents:** Uploader, Verification, and Approval Agents handle distinct workflow stages.
  * **Model Context Protocol (MCP):** A mock FastAPI server (`http://localhost:8900`) serves as a central source of truth for all workflow state updates, mimicking a database or state service.
  * **Automated Risk Assessment:** Contracts are categorized into low, medium, or high risk based on content analysis.
  * **Conditional Routing:** Automatically routes low-risk contracts to the `auto_approved` state, bypassing manual review, but still ensures a **digital signature** is created (the core fix implemented).
  * **Robust PDF Handling:** Uses `PyPDF2` with error handling for text extraction.
  * **FastAPI Web Interface:** Provides a simple, interactive HTML front-end for testing the entire workflow (upload, view status, manual approval).

-----

## 🏗️ Architecture

The system runs as two concurrent services (managed by `asyncio.gather`):

1.  **MCP Server (Port 8900):** Stores the `ContractWorkflowState` for all active workflows. Agents use the `update_mcp_context` tool to persist changes.
2.  **Orchestrator (Port 8000):** Hosts the FastAPI application and runs the LangGraph engine (`ContractApprovalWorkflow`).

**Workflow Flow:**

`Uploader` ➡️ `Supervisor` ➡️ `Verification` ➡️ `Supervisor` ➡️ `Approval` ➡️ **END**

  * **Uploader:** Extracts text from the PDF.
  * **Verification:** Analyzes text, assesses risk (e.g., 60% risk), and routes the workflow:
      * **Low Risk (`<= 30%`):** Sets `approval_decision` to `auto_approved` and routes to `Approval`.
      * **Medium/High Risk (`> 30%`):** Sets status to `pending_approval` and routes to `Approval`.
  * **Approval:** Processes any existing decision (auto-approved) or waits for manual input. If the final decision is **`approved`** (manual or auto), it invokes the **`create_digital_signature`** tool before transitioning to **END**.

-----

##  Prerequisites

  * **Python 3.9+**
  * The following libraries:
    ```bash
    pip install langgraph langchain-core langchain-openai fastapi uvicorn pydantic httpx PyPDF2
    ```
    *(Note: `langchain-openai` is used for the `ChatOpenAI` class, which is mocked in this MVP. The mock does not require a real OpenAI API key.)*

-----

##  Installation and Usage

### 1\. Save the Code

Save the entire provided code block into a single file named `complete_mvp_langgraph.py`.

### 2\. Run the System

Execute the script from your terminal:

```bash
python complete_mvp_langgraph.py
```

This command starts both the **MCP Server** (port 8900) and the **Orchestrator** (port 8000) simultaneously.

### 3\. Access the Web Interface

Open your web browser and navigate to:

 **[http://localhost:8000](https://www.google.com/search?q=http://localhost:8000)**

-----

##  Testing the Workflow

Use the provided web interface to test both auto-approval and manual approval paths.

1.  **Login:** Use the mock username `demo` (has `approve` permission).
2.  **Upload:** Use any PDF file. The risk assessment logic is intentionally simple/mocked to generate a score based on length and clause presence.
3.  **Observe:**
      * **Low Risk Path (Auto-Approval):** If your PDF is short and clean, it will quickly progress to `auto_approved`, and the `Approval Decision` will show a **`Signature Hash`**.
      * **Medium/High Risk Path (Manual Approval):** If the analysis triggers medium/high risk, the workflow will stop at **`PENDING APPROVAL`**. Use the action buttons to manually **Approve** (which generates the signature) or **Reject** (which skips the signature).

-----

##  Key Fix Implemented

The initial workflow design bypassed the `approval` agent for low-risk contracts, leading to **missing digital signatures** for auto-approved documents.

The fix ensures that:

1.  The **`ContractVerificationAgent`** now routes **all** final decisions (auto-approved or pending) to the **`ApprovalAgent`**.
2.  The **`ContractApprovalAgent`** checks if the decision is already set (auto-approved). If the decision is `approved` and the `signature_hash` is missing, it is immediately generated, ensuring all final approved contracts are signed.