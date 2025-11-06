"""
Contract Approval System MVP - LangGraph Agents + MCP + Orchestration
Demonstrates: LangGraph multi-agent system with MCP coordination

This file includes robust error handling in the PDF extraction step and
graceful error propagation within the Uploader Agent.
"""

import asyncio
import json
import uuid
import hashlib
import re
import time
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, TypedDict, Annotated
from dataclasses import dataclass, asdict
from enum import Enum
import operator
import sys
import os # Added for better system checks

# Core dependencies
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import httpx
import PyPDF2
from io import BytesIO

# LangGraph dependencies
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Mock OpenAI for MVP (replace with real OpenAI in production)
class MockOpenAI:
    """Mock OpenAI for MVP demonstration"""
    
    def __init__(self, model="gpt-4"):
        self.model = model
    
    async def ainvoke(self, messages):
        """Mock AI response based on message context"""
        last_message = messages[-1].content if messages else ""
        
        # Check if the error has been explicitly set in the state before responding
        if "Error:" in last_message:
            return AIMessage(content=f"Workflow failed due to contract processing error. Details: {last_message}")
        
        if "upload" in last_message.lower():
            return AIMessage(content="Contract uploaded successfully. Initiating verification process.")
        elif "analyze" in last_message.lower() or "verify" in last_message.lower():
            return AIMessage(content="Contract analysis complete. Found payment terms, liability clauses. Risk assessment: Medium risk due to potential issues.")
        elif "approve" in last_message.lower():
            return AIMessage(content="Contract approval processed. Digital signature created.")
        else:
            return AIMessage(content="Processing request...")

# Replace ChatOpenAI with MockOpenAI for MVP
ChatOpenAI = MockOpenAI

# =============================================================================
# CONFIGURATION & STATE MANAGEMENT
# =============================================================================

# Environment configuration
CONFIG = {
    "openai_api_key": "mock-key-for-mvp",  # Use real key in production
    "descope_project_id": "your-descope-project-id",
    "mcp_port": 8900,
    "orchestrator_port": 8000,
}

# Mock users for MVP
MOCK_USERS = {
    "demo": {"user_id": "user_123", "role": "manager", "permissions": ["upload", "approve"]},
    "analyst": {"user_id": "user_456", "role": "analyst", "permissions": ["upload"]},
}

# LangGraph State Definition
class ContractWorkflowState(TypedDict):
    """Shared state between all agents in the workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    workflow_id: str
    user_id: str
    contract_data: Dict[str, Any]
    extracted_text: str
    analysis_results: Dict[str, Any]
    approval_decision: Optional[Dict[str, Any]]
    current_step: str
    next_agent: str
    error_message: Optional[str]
    processing_metadata: Dict[str, Any]

class WorkflowStatus(Enum):
    CREATED = "created"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ERROR = "error"

# =============================================================================
# LANGCHAIN TOOLS FOR AGENTS
# =============================================================================

@tool
def extract_pdf_text(pdf_content_base64: str) -> str:
    """Extract text from PDF file with robust error handling"""
    try:
        # Decode base64 content
        content = base64.b64decode(pdf_content_base64)
        pdf_file = BytesIO(content)
        
        # Try to open PDF with PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
        except Exception as e:
            # Catch specific PyPDF2 issues like malformed files
            return f"Error: Failed to open PDF file with PyPDF2. The PDF may be corrupted or in an unsupported format. Details: {str(e)}"
        
        # Extract text from all pages
        text_parts = []
        total_pages = len(pdf_reader.pages)
        
        print(f"Processing PDF with {total_pages} pages...")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
                    print(f"Extracted {len(page_text)} characters from page {page_num + 1}")
            except Exception as e:
                # Log that a specific page failed but continue
                print(f"Warning: Failed to extract text from page {page_num + 1}. Error: {str(e)}")
                continue
        
        # Combine all text
        if not text_parts:
            return "Error: No text could be extracted from the PDF. The PDF may contain only images or be encrypted."
        
        full_text = "\n".join(text_parts)
        
        # Clean up the text
        full_text = full_text.replace('\x00', '')  # Remove null bytes
        full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
        full_text = full_text.strip()
        
        print(f"Successfully extracted {len(full_text)} characters total")
        
        return full_text
        
    except Exception as e:
        # Catch all other errors (e.g., base64 decoding)
        error_msg = f"Fatal Error during PDF processing: {str(e)}"
        print(error_msg)
        return error_msg

@tool
def analyze_contract_clauses(contract_text: str) -> Dict[str, Any]:
    """Analyze contract text to extract clauses and assess risk"""
    clause_patterns = {
        "payment_terms": [r"payment.*?(\d+)\s*days?", r"net\s*(\d+)", r"invoice.*?(\d+)\s*days?"],
        "termination": [r"terminat.*?notice", r"end.*?agreement", r"breach.*?contract"],
        "liability": [r"liability.*?limited?", r"damages.*?exceed", r"indemnif"],
        "confidentiality": [r"confidential.*?information", r"non-disclosure", r"proprietary"]
    }
    
    clauses = []
    risk_factors = []
    
    # Extract clauses
    for clause_type, patterns in clause_patterns.items():
        found = False
        for pattern in patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(contract_text), match.end() + 30)
                context = contract_text[start:end].replace('\n', ' ').strip()
                clauses.append({
                    "type": clause_type,
                    "content": context,
                    "confidence": 0.85
                })
                found = True
                break
            if found:
                break
    
    # Risk assessment
    found_clause_types = {clause["type"] for clause in clauses}
    critical_clauses = {"payment_terms", "termination", "liability"}
    
    risk_score = 0.0
    for critical in critical_clauses:
        if critical not in found_clause_types:
            risk_score += 0.3
            risk_factors.append(f"Missing {critical.replace('_', ' ')} clause")
    
    if len(contract_text) < 500:
        risk_score += 0.2
        risk_factors.append("Contract appears too brief")
    
    risk_level = "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.4 else "low"
    
    return {
        "clauses": clauses,
        "risk_assessment": {
            "overall_risk": min(risk_score, 1.0),
            "risk_level": risk_level,
            "risk_factors": risk_factors
        },
        "word_count": len(contract_text.split()),
        "analysis_timestamp": datetime.utcnow().isoformat()
    }

@tool
def create_digital_signature(workflow_id: str, decision: str, user_id: str) -> str:
    """Create digital signature for contract approval"""
    timestamp = datetime.utcnow().isoformat()
    content = f"{workflow_id}:{decision}:{user_id}:{timestamp}"
    signature_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return signature_hash

@tool
def update_mcp_context(workflow_id: str, updates: Dict[str, Any], mcp_url: str = "http://localhost:8900") -> str:
    """Update workflow context in MCP server"""
    try:
        # In MVP, we'll simulate MCP update
        return f"Updated context for workflow {workflow_id} with {len(updates)} fields"
    except Exception as e:
        return f"Error updating MCP context: {str(e)}"

# =============================================================================
# LANGRAPH AGENTS
# =============================================================================

class ContractUploaderAgent:
    """LangGraph Agent A: Contract Upload Handler"""
    
    def __init__(self):
        self.name = "UploaderAgent"
        self.llm = ChatOpenAI(model="gpt-4")
        self.tools = [extract_pdf_text, update_mcp_context]
    
    async def process(self, state: ContractWorkflowState) -> ContractWorkflowState:
        """Process contract upload"""
        
        system_prompt = """You are a Contract Upload Agent. Your responsibilities:
        1. Receive and validate contract documents
        2. Extract text content from PDF files
        3. Store contract metadata
        4. Coordinate with verification agent
        
        Always be professional and thorough in your analysis."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Process contract upload for workflow {state['workflow_id']}. Extract text and prepare for analysis.")
        ]
        
        # Process contract data if available
        if state.get("contract_data") and state["contract_data"].get("content_base64"):
            # Extract text using tool
            extracted_text = extract_pdf_text.invoke(state["contract_data"]["content_base64"])
            
            # --- NEW ERROR HANDLING LOGIC ---
            if extracted_text.startswith("Error:"):
                # If the tool returned an error string, set the state to ERROR
                state["error_message"] = extracted_text
                state["current_step"] = "error"
                state["next_agent"] = "error" # Route to the END state
                
                # Update MCP context with the error status
                update_mcp_context.invoke({
                    "workflow_id": state["workflow_id"],
                    "updates": {"status": "error", "error_message": extracted_text},
                    "mcp_url": "http://localhost:8900"
                })
                
                # Add the error to messages for the LLM to acknowledge (optional, but useful)
                messages.append(AIMessage(content=f"FATAL ERROR during extraction: {extracted_text}"))
            
            else:
                # Normal successful path
                state["extracted_text"] = extracted_text
                state["current_step"] = "text_extracted"
                state["next_agent"] = "verification"
                
                # Update MCP context
                update_mcp_context.invoke({
                    "workflow_id": state["workflow_id"],
                    "updates": {"extracted_text": extracted_text[:200], "status": "analyzing"},
                    "mcp_url": "http://localhost:8900"
                })
        else:
             # Should not happen in this flow, but handles missing data
            state["error_message"] = "Contract content not found in state."
            state["current_step"] = "error"
            state["next_agent"] = "error"
        
        # Get AI response (if not already set to error)
        if state["current_step"] != "error":
            response = await self.llm.ainvoke(messages)
            state["messages"].append(response)
            
        state["processing_metadata"]["uploader_completed"] = datetime.utcnow().isoformat()
        
        return state

class ContractVerificationAgent:
    """LangGraph Agent B: Contract Analysis and Verification"""
    
    def __init__(self):
        self.name = "VerificationAgent"
        self.llm = ChatOpenAI(model="gpt-4")
        self.tools = [analyze_contract_clauses, update_mcp_context]
    
    async def process(self, state: ContractWorkflowState) -> ContractWorkflowState:
        """Process contract verification and analysis"""
        
        system_prompt = """You are a Contract Verification Agent with expertise in legal document analysis. Your responsibilities:
        1. Analyze contract text for key clauses (payment, termination, liability, confidentiality)
        2. Assess risk factors and overall contract risk level
        3. Identify missing critical clauses
        4. Provide recommendations for approval/rejection
        
        Be thorough and identify potential legal issues."""
        
        contract_text = state.get("extracted_text", "")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze the following contract text and provide detailed analysis:\n\n{contract_text[:500]}...")
        ]
        
        # Get AI response
        response = await self.llm.ainvoke(messages)
        
        # Analyze contract using tool
        if contract_text:
            analysis_results = analyze_contract_clauses.invoke(contract_text)
            state["analysis_results"] = analysis_results
            
            # Determine next step based on risk
            risk_level = analysis_results.get("risk_assessment", {}).get("overall_risk", 0.5)
            
            # --- FIX: Route all non-error paths to 'approval' agent (manual or auto)
            if risk_level > 0.3:
                # Manual approval path (High/Medium risk)
                state["next_agent"] = "approval"
                state["current_step"] = "pending_approval"
                
                # Update MCP context
                update_mcp_context.invoke({
                    "workflow_id": state["workflow_id"],
                    "updates": {"analysis_results": analysis_results, "status": state["current_step"]},
                    "mcp_url": "http://localhost:8900"
                })

            else:
                # Auto-approval path (Low risk)
                state["current_step"] = "auto_approved"
                state["next_agent"] = "approval" # <--- FIXED: Route to 'approval' agent for signing
                state["approval_decision"] = {
                    "decision": "approved",
                    "reason": "auto_approved_low_risk",
                    "timestamp": datetime.utcnow().isoformat(),
                    "approved_by": "system"
                }
                
                # Update MCP context (send auto-approved status and decision to MCP)
                update_mcp_context.invoke({
                    "workflow_id": state["workflow_id"],
                    "updates": {"analysis_results": analysis_results, "status": state["current_step"], "approval_decision": state["approval_decision"]},
                    "mcp_url": "http://localhost:8900"
                })
        
        # Add messages
        state["messages"].append(response)
        state["processing_metadata"]["verification_completed"] = datetime.utcnow().isoformat()
        
        return state

class ContractApprovalAgent:
    """LangGraph Agent C: Contract Approval and Signing"""
    
    def __init__(self):
        self.name = "ApprovalAgent"
        self.llm = ChatOpenAI(model="gpt-4")
        self.tools = [create_digital_signature, update_mcp_context]
    
    async def process(self, state: ContractWorkflowState) -> ContractWorkflowState:
        """Process contract approval decision"""
        
        system_prompt = """You are a Contract Approval Agent responsible for final approval decisions. Your responsibilities:
        1. Review contract analysis and risk assessment
        3. Create digital signatures for approved contracts
        4. Maintain audit trails
        
        Ensure all approvals are properly documented and signed."""
        
        analysis_results = state.get("analysis_results", {})
        risk_info = analysis_results.get("risk_assessment", {})
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Process approval for workflow {state['workflow_id']}. Risk level: {risk_info.get('risk_level', 'unknown')}")
        ]
        
        # Get AI response
        response = await self.llm.ainvoke(messages)
        
        # If approval decision is already made (either manual or auto), process it
        if state.get("approval_decision"):
            decision = state["approval_decision"]
            
            # Create digital signature if approved AND signature is missing
            if decision.get("decision") == "approved" and not decision.get("signature_hash"):
                signature = create_digital_signature.invoke({
                    "workflow_id": state["workflow_id"],
                    "decision": decision["decision"],
                    "user_id": decision.get("approved_by", "system")
                })
                decision["signature_hash"] = signature
            
            # Update status for final state
            if decision["decision"] == "approved":
                state["current_step"] = "approved"
            elif decision["decision"] == "rejected":
                state["current_step"] = "rejected"
            else:
                 # Should only happen if coming from auto_approved and has decision
                 state["current_step"] = "approved" # Treat auto_approved as final approved state

            state["next_agent"] = "end" # <--- Route to END
            
            # Update MCP context
            update_mcp_context.invoke({
                "workflow_id": state["workflow_id"],
                "updates": {"approval_decision": decision, "status": state["current_step"]},
                "mcp_url": "http://localhost:8900"
            })
        
        # Add messages
        state["messages"].append(response)
        state["processing_metadata"]["approval_completed"] = datetime.utcnow().isoformat()
        
        return state

# =============================================================================
# LANGRAPH WORKFLOW DEFINITION
# =============================================================================

class ContractApprovalWorkflow:
    """LangGraph workflow orchestrating all agents"""
    
    def __init__(self):
        self.uploader_agent = ContractUploaderAgent()
        self.verification_agent = ContractVerificationAgent()
        self.approval_agent = ContractApprovalAgent()
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the graph
        workflow = StateGraph(ContractWorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("uploader", self.uploader_node)
        workflow.add_node("verification", self.verification_node)
        workflow.add_node("approval", self.approval_node)
        workflow.add_node("supervisor", self.supervisor_node)
        
        # Define edges
        workflow.set_entry_point("uploader")
        workflow.add_edge("uploader", "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self.route_next_agent,
            {
                "verification": "verification",
                "approval": "approval",
                "completed": END,
                "error": END # Added explicit error path to END
            }
        )
        workflow.add_edge("verification", "supervisor")
        workflow.add_edge("approval", END) # <--- Approval now always goes to END
        
        return workflow.compile()
    
    async def uploader_node(self, state: ContractWorkflowState) -> ContractWorkflowState:
        """Execute uploader agent"""
        return await self.uploader_agent.process(state)
    
    async def verification_node(self, state: ContractWorkflowState) -> ContractWorkflowState:
        """Execute verification agent"""
        return await self.verification_agent.process(state)
    
    async def approval_node(self, state: ContractWorkflowState) -> ContractWorkflowState:
        """Execute approval agent"""
        return await self.approval_agent.process(state)
    
    async def supervisor_node(self, state: ContractWorkflowState) -> ContractWorkflowState:
        """Supervisor node to determine next action"""
        
        current_step = state.get("current_step", "")
        next_agent = state.get("next_agent", "")
        
        # Update supervisor decision
        state["processing_metadata"]["supervisor_decision"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_step": current_step,
            "next_agent": next_agent
        }
        
        return state
    
    def route_next_agent(self, state: ContractWorkflowState) -> str:
        """Routing function to determine next agent"""
        next_agent = state.get("next_agent", "completed")
        
        # If the agent explicitly set the next step to 'error', route to END
        if next_agent == "error":
            return "error"

        if next_agent in ["verification", "approval", "completed"]:
            # 'completed' means the workflow ends, but this is handled by END in the graph
            return next_agent
        else:
            return "completed" # Default to completing the workflow
    
    async def run_workflow(self, initial_state: ContractWorkflowState) -> ContractWorkflowState:
        """Execute the complete workflow"""
        try:
            # We use ainvoke here which should execute the LangGraph, 
            # including the error handling added in the UploaderAgent.
            result = await self.workflow.ainvoke(initial_state)
            return result
        except Exception as e:
            # This catches exceptions that bubble up outside the LangGraph flow
            # (e.g., configuration errors, network issues).
            initial_state["error_message"] = f"Uncaught LangGraph Engine Error: {str(e)}"
            initial_state["current_step"] = "error"
            return initial_state

# =============================================================================
# MCP SERVER
# =============================================================================

class MCPServer:
    """Model Context Protocol Server for agent coordination"""
    
    def __init__(self, port: int = 8900):
        self.port = port
        self.app = FastAPI(title="MCP Server", version="1.0.0")
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/mcp/invoke")
        async def invoke_method(request: dict):
            """Handle MCP method invocations"""
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "get_context":
                return await self.get_context(params)
            elif method == "update_context":
                return await self.update_context(params)
            elif method == "create_context":
                return await self.create_context(params)
            else:
                return {"error": f"Unknown method: {method}"}
        
        @self.app.get("/mcp/contexts")
        async def list_contexts():
            """List all contexts"""
            return {"contexts": list(self.contexts.values())}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "mcp-server"}
    
    async def create_context(self, params: dict) -> dict:
        """Create new workflow context"""
        workflow_id = params.get("workflow_id")
        user_id = params.get("user_id")
        
        context = {
            "workflow_id": workflow_id,
            "user_id": user_id,
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "contract_data": {},
            "analysis_results": {},
            "approval_decision": None
        }
        
        self.contexts[workflow_id] = context
        return {"success": True, "context": context}
    
    async def get_context(self, params: dict) -> dict:
        """Get workflow context"""
        workflow_id = params.get("workflow_id")
        if workflow_id in self.contexts:
            return {"success": True, "context": self.contexts[workflow_id]}
        else:
            return {"success": False, "error": "Context not found"}
    
    async def update_context(self, params: dict) -> dict:
        """Update workflow context"""
        workflow_id = params.get("workflow_id")
        updates = params.get("updates", {})
        
        if workflow_id not in self.contexts:
            return {"success": False, "error": "Context not found"}
        
        context = self.contexts[workflow_id]
        
        # Special handling for dictionary updates (like analysis_results)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(context.get(key), dict):
                context[key].update(value)
            else:
                context[key] = value
                
        context["updated_at"] = datetime.utcnow().isoformat()
        
        return {"success": True, "context": context}

# =============================================================================
# DESCOPE AUTHENTICATION (Mock)
# =============================================================================

class DescopeAuth:
    """Mock Descope authentication for MVP"""
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate user token"""
        if token in MOCK_USERS:
            user = MOCK_USERS[token]
            return {"valid": True, **user}
        return {"valid": False, "error": "Invalid token"}
    
    async def create_agent_token(self, user_id: str, scopes: List[str]) -> str:
        """Create agent-to-agent token"""
        token_data = {
            "user_id": user_id,
            "scopes": scopes,
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        return base64.b64encode(json.dumps(token_data).encode()).decode()

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class ContractApprovalOrchestrator:
    """Main orchestrator using LangGraph workflow"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.app = FastAPI(title="Contract Approval Orchestrator", version="1.0.0")
        self.workflow_engine = ContractApprovalWorkflow()
        self.mcp_server_url = f"http://localhost:{CONFIG['mcp_port']}"
        self.auth = DescopeAuth()
        self.active_workflows: Dict[str, ContractWorkflowState] = {}
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            return self.get_web_interface()
        
        @self.app.post("/login")
        async def login(credentials: dict):
            """User authentication"""
            username = credentials.get("username")
            password = credentials.get("password")  # In MVP, password is ignored
            
            if username in MOCK_USERS:
                return {
                    "success": True,
                    "token": username,
                    "user": MOCK_USERS[username]
                }
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        @self.app.post("/upload-contract")
        async def upload_contract(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            user_token: str = Form(...)
        ):
            """Orchestrate contract upload using LangGraph workflow"""
            
            # Validate user
            user_validation = await self.auth.validate_token(user_token)
            if not user_validation.get("valid"):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            user_id = user_validation["user_id"]
            workflow_id = str(uuid.uuid4())[:8]
            
            try:
                # Read file content
                content = await file.read()
                content_base64 = base64.b64encode(content).decode()
                
                # Create initial state
                initial_state: ContractWorkflowState = {
                    "messages": [HumanMessage(content=f"Process contract upload for {file.filename}")],
                    "workflow_id": workflow_id,
                    "user_id": user_id,
                    "contract_data": {
                        "filename": file.filename,
                        "size": len(content),
                        "content_base64": content_base64,
                        "uploaded_at": datetime.utcnow().isoformat()
                    },
                    "extracted_text": "",
                    "analysis_results": {},
                    "approval_decision": None,
                    "current_step": "uploading",
                    "next_agent": "uploader",
                    "error_message": None,
                    "processing_metadata": {"started_at": datetime.utcnow().isoformat()}
                }
                
                # Store workflow state
                self.active_workflows[workflow_id] = initial_state
                
                # Process workflow in background
                background_tasks.add_task(self.process_workflow, workflow_id, initial_state)
                
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "message": "Contract uploaded successfully, processing started"
                }
                
            except Exception as e:
                # Catch upload/file read error here
                return {"success": False, "error": str(e)}
        
        @self.app.post("/approve-contract")
        async def approve_contract(request: dict):
            """Process manual approval decision"""
            user_token = request.get("user_token")
            workflow_id = request.get("workflow_id")
            decision = request.get("decision")  # "approve" or "reject"
            reason = request.get("reason", "")
            
            # Validate user
            user_validation = await self.auth.validate_token(user_token)
            if not user_validation.get("valid"):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            if workflow_id not in self.active_workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            # Update workflow with approval decision
            workflow_state = self.active_workflows[workflow_id]
            workflow_state["approval_decision"] = {
                "decision": decision,
                "reason": reason,
                "approved_by": user_validation["user_id"],
                "timestamp": datetime.utcnow().isoformat()
            }
            workflow_state["current_step"] = "processing_approval"
            workflow_state["next_agent"] = "approval"
            
            # Continue workflow processing
            asyncio.create_task(self.continue_workflow(workflow_id))
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "message": f"Contract {decision} decision recorded"
            }
        
        @self.app.get("/workflows")
        async def list_workflows():
            """Get all active workflows"""
            workflows = []
            for wf_id, state in self.active_workflows.items():
                workflows.append({
                    "workflow_id": wf_id,
                    "status": state.get("current_step", "unknown"),
                    "user_id": state.get("user_id"),
                    "created_at": state.get("processing_metadata", {}).get("started_at"),
                    "contract_filename": state.get("contract_data", {}).get("filename"),
                    "analysis_results": state.get("analysis_results"),
                    "approval_decision": state.get("approval_decision"),
                    "error_message": state.get("error_message")
                })
            return {"workflows": workflows}
        
        @self.app.get("/workflow/{workflow_id}")
        async def get_workflow(workflow_id: str):
            """Get specific workflow details"""
            if workflow_id not in self.active_workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            state = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "current_step": state.get("current_step"),
                "error_message": state.get("error_message"), # Added error message here
                "contract_data": state.get("contract_data"),
                "analysis_results": state.get("analysis_results"),
                "approval_decision": state.get("approval_decision"),
                "processing_metadata": state.get("processing_metadata"),
                "messages": [{"content": msg.content, "type": type(msg).__name__} for msg in state.get("messages", [])]
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "active_workflows": len(self.active_workflows),
                "timestamp": datetime.utcnow().isoformat()
            }
        
    async def process_workflow(self, workflow_id: str, initial_state: ContractWorkflowState):
        """Process workflow using LangGraph"""
        try:
            # Run the LangGraph workflow
            final_state = await self.workflow_engine.run_workflow(initial_state)
            
            # Update stored state
            self.active_workflows[workflow_id] = final_state
            
            # Log completion
            print(f"Workflow {workflow_id} completed with status: {final_state.get('current_step')}")
            
        except Exception as e:
            # This handles unexpected LangGraph failures outside a node's try/except block
            print(f"Workflow {workflow_id} failed with uncaught exception: {str(e)}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["error_message"] = f"Uncaught Workflow Error: {str(e)}"
                self.active_workflows[workflow_id]["current_step"] = "error"
    
    async def continue_workflow(self, workflow_id: str):
        """Continue workflow processing after approval decision"""
        if workflow_id in self.active_workflows:
            state = self.active_workflows[workflow_id]
            # Run approval agent
            updated_state = await self.workflow_engine.approval_agent.process(state)
            self.active_workflows[workflow_id] = updated_state
            
    def get_web_interface(self) -> str:
        """Complete web interface for testing"""
        return """
<!DOCTYPE html>
<html>
<head>
<title>Contract Approval System - LangGraph MVP</title>
<style>
body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; } 
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; } 
.section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); } 
.status { padding: 12px; margin: 10px 0; border-radius: 5px; font-weight: bold; } 
.status.uploading { background-color: #fff3cd; color: #856404; } 
.status.analyzing { background-color: #d1ecf1; color: #0c5460; } 
.status.pending_approval { background-color: #f8d7da; color: #721c24; } 
.status.completed, .status.approved, .status.auto_approved { background-color: #d4edda; color: #155724; }
.status.rejected, .status.error { background-color: #f8d7da; color: #721c24; } 
.agent-flow { display: flex; justify-content: space-around; margin-top: 15px; } 
.agent-box { background: rgba(255, 255, 255, 0.1); padding: 10px 15px; border-radius: 5px; text-align: center; } 
input[type="text"], input[type="password"], input[type="file"], textarea { width: 100%; padding: 10px; margin: 8px 0; display: inline-block; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; } 
button { background-color: #667eea; color: white; padding: 10px 15px; margin: 8px 0; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; } 
button:hover:not(:disabled) { background-color: #556ee0; } 
button:disabled { background-color: #aaa; cursor: not-allowed; } 
.hidden { display: none; } 
.workflow-card { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; } 
.risk-indicator { padding: 5px 10px; border-radius: 3px; font-weight: bold; } 
.risk-low { background-color: #d4edda; color: #155724; } 
.risk-medium { background-color: #fff3cd; color: #856404; } 
.risk-high { background-color: #f8d7da; color: #721c24; } 
.spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s ease-in-out infinite; } 
@keyframes spin { to { transform: rotate(360deg); } } 
.footer { margin-top: 40px; text-align: center; color: #666; font-size: 14px; } 
</style>
</head>
<body>

<div class="header">
    <h1>🤖 Contract Approval System</h1>
    <p>LangGraph Multi-Agent Workflow with MCP Coordination</p>
    <div class="agent-flow">
        <div class="agent-box">
            <strong>📄 Uploader Agent</strong><br>
            <small>PDF Processing & Text Extraction</small>
        </div>
        <div class="agent-box">
            <strong>🔍 Verification Agent</strong><br>
            <small>Contract Analysis & Risk Assessment</small>
        </div>
        <div class="agent-box">
            <strong>✅ Approval Agent</strong><br>
            <small>Decision Processing & Digital Signing</small>
        </div>
    </div>
</div>

<div class="section login-section">
    <h2>🔐 Authentication</h2>
    <div id="login-form">
        <input type="text" id="username" placeholder="Username (demo/analyst)" value="demo">
        <input type="password" id="password" placeholder="Password (any)">
        <button onclick="login()">Login</button>
    </div>
    <div id="user-info" class="hidden">
        <p>Logged in as: <span id="user-display"></span></p>
        <button onclick="logout()">Logout</button>
    </div>
</div>

<div class="section upload-section">
    <h2>📤 Upload Contract</h2>
    <input type="file" id="contract-file" accept=".pdf" disabled>
    <button onclick="uploadContract()" id="upload-btn" disabled>Upload & Process</button>
    <div id="upload-status"></div>
</div>

<div class="section workflows-section">
    <h2>🔄 Active Workflows</h2>
    <button onclick="refreshWorkflows()">🔄 Refresh</button>
    <div id="workflows-container">
        <p>No workflows yet. Upload a contract to start!</p>
    </div>
</div>

<div class="section status-section">
    <h2>📊 System Status</h2>
    <div id="system-status">
        Loading status...
    </div>
</div>

<div class="footer">
    <p>Powered by LangGraph, FastAPI, and MCP (Mock)</p>
</div>

<script>
    const USERS = {
        "demo": {"user_id": "user_123", "role": "manager", "permissions": ["upload", "approve"]},
        "analyst": {"user_id": "user_456", "role": "analyst", "permissions": ["upload"]},
    };

    let userToken = localStorage.getItem('userToken');
    let currentUser = JSON.parse(localStorage.getItem('currentUser'));

    function showStatus(elementId, message, type) {
        const statusElement = document.getElementById(elementId);
        statusElement.innerHTML = `<span class="status ${type}">${message}</span>`;
    }

    function updateUI() {
        const loginForm = document.getElementById('login-form');
        const userInfo = document.getElementById('user-info');
        const fileInput = document.getElementById('contract-file');
        const uploadBtn = document.getElementById('upload-btn');
        const userDisplay = document.getElementById('user-display');

        if (userToken && currentUser) {
            loginForm.classList.add('hidden');
            userInfo.classList.remove('hidden');
            userDisplay.textContent = `${currentUser.user_id} (${currentUser.role})`;
            
            // Enable upload if logged in
            fileInput.disabled = false;
            uploadBtn.disabled = false;
        } else {
            loginForm.classList.remove('hidden');
            userInfo.classList.add('hidden');
            fileInput.disabled = true;
            uploadBtn.disabled = true;
        }
        refreshWorkflows();
        updateSystemStatus();
    }

    // Login
    async function login() {
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value; // Mock: not used

        if (username in USERS) {
            userToken = username;
            currentUser = USERS[username];
            localStorage.setItem('userToken', userToken);
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            showStatus('upload-status', 'Login successful!', 'approved');
            updateUI();
        } else {
            showStatus('upload-status', 'Invalid credentials', 'error');
        }
    }

    // Logout
    function logout() {
        userToken = null;
        currentUser = null;
        localStorage.removeItem('userToken');
        localStorage.removeItem('currentUser');
        showStatus('upload-status', 'analyzing', 'analyzing');
        updateUI();
    }

    // Upload contract
    async function uploadContract() {
        if (!userToken) {
            alert('Please login first.');
            return;
        }
        
        const fileInput = document.getElementById('contract-file');
        if (fileInput.files.length === 0) {
            alert('Please select a PDF file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('user_token', userToken);

        showStatus('upload-status', 'Uploading and processing contract...', 'uploading');

        try {
            const response = await fetch('/upload-contract', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (result.success) {
                showStatus('upload-status', `Contract uploaded! Workflow ID: ${result.workflow_id}`, 'approved');
                // Auto-refresh workflows after 2 seconds
                setTimeout(refreshWorkflows, 2000);
            } else {
                showStatus('upload-status', 'Upload failed: ' + result.error, 'error');
            }
        } catch (error) {
            showStatus('upload-status', 'Upload error: ' + error.message, 'error');
        }
    }

    // Refresh workflows
    async function refreshWorkflows() {
        try {
            const response = await fetch('/workflows');
            const result = await response.json();
            const container = document.getElementById('workflows-container');

            if (result.workflows.length === 0) {
                container.innerHTML = '<p>No workflows yet. Upload a contract to start!</p>';
                return;
            }

            let html = '';
            result.workflows.forEach(workflow => {
                const riskLevel = workflow.analysis_results?.risk_assessment?.risk_level || 'unknown';
                const riskClass = `risk-${riskLevel}`;
                const isPending = workflow.status === 'pending_approval';
                const isError = workflow.status === 'error';
                const isApproved = workflow.status === 'approved' || workflow.status === 'auto_approved';

                html += `
                    <div class="workflow-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3>📄 ${workflow.contract_filename || 'Unknown'}</h3>
                            <span class="risk-indicator ${isError ? 'risk-high' : riskClass}">${isError ? 'ERROR' : 'Risk: ' + riskLevel.toUpperCase()}</span>
                        </div>
                        <p><strong>Workflow ID:</strong> ${workflow.workflow_id}</p>
                        <p><strong>Status:</strong> <span class="status ${workflow.status}">${workflow.status.toUpperCase().replace('_', ' ')}</span></p>
                        <p><strong>User:</strong> ${workflow.user_id}</p>
                        <p><strong>Created:</strong> ${new Date(workflow.created_at).toLocaleString()}</p>

                        ${isError ? `
                        <div style="margin: 10px 0; padding: 10px; background: #f8d7da; border-radius: 5px; color: #721c24; font-weight: bold;">
                            <strong>Error Message:</strong> ${workflow.error_message || 'Unknown processing error.'}
                        </div>
                        ` : ''}
                        
                        ${workflow.analysis_results && !isError ? `
                        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                            <strong>Analysis Results:</strong><br>
                            Risk Score: ${(workflow.analysis_results.risk_assessment?.overall_risk * 100).toFixed(1)}%
                            <br>Risk Factors: ${workflow.analysis_results.risk_assessment?.risk_factors.join(', ') || 'None'}
                        </div>
                        ` : ''}

                        ${workflow.approval_decision ? `
                        <div style="margin: 10px 0; padding: 10px; border-radius: 5px; background: ${workflow.approval_decision.decision === 'approved' ? '#d4edda' : '#f8d7da'};">
                            <strong>Final Decision:</strong> ${workflow.approval_decision.decision.toUpperCase()} by ${workflow.approval_decision.approved_by}
                            ${workflow.approval_decision.signature_hash ? `<br>Signature: <small>${workflow.approval_decision.signature_hash}</small>` : '<br>Signature: <span style="font-weight: bold; color: #dc3545;">MISSING</span>'}
                        </div>
                        ` : ''}

                        <button onclick="viewWorkflowDetails('${workflow.workflow_id}')">View Details</button>
                        
                        ${isPending && currentUser?.permissions.includes('approve') ? `
                        <div style="margin-top: 15px; padding-top: 10px; border-top: 1px dashed #eee;">
                            <strong>Approval Action:</strong>
                            <input type="text" id="reason-${workflow.workflow_id}" placeholder="Reason for approval/rejection">
                            <button onclick="processApproval('${workflow.workflow_id}', 'approved')" style="background-color: #28a745;">✅ Approve</button>
                            <button onclick="processApproval('${workflow.workflow_id}', 'rejected')" style="background-color: #dc3545;">❌ Reject</button>
                        </div>
                        ` : ''}
                    </div>
                `;
            });
            container.innerHTML = html;

        } catch (error) {
            container.innerHTML = '<p>Error loading workflows: ' + error.message + '</p>';
        }
    }
    
    // Process manual approval
    async function processApproval(workflowId, decision) {
        const reasonInput = document.getElementById(`reason-${workflowId}`);
        const reason = reasonInput ? reasonInput.value : '';

        if (!reason) {
            alert('Please provide a reason for the decision.');
            return;
        }

        try {
            const response = await fetch('/approve-contract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_token: userToken,
                    workflow_id: workflowId,
                    decision: decision,
                    reason: reason
                })
            });
            const result = await response.json();

            if (result.success) {
                alert(`Contract ${decision} successfully! Workflow will finalize shortly.`);
                setTimeout(refreshWorkflows, 1000);
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }

    // View workflow details
    async function viewWorkflowDetails(workflowId) {
        try {
            const response = await fetch(`/workflow/${workflowId}`);
            const workflow = await response.json();

            let details = `
WORKFLOW DETAILS
================
Workflow ID: ${workflow.workflow_id}
Current Step: ${workflow.current_step}
Error Message: ${workflow.error_message || 'None'}

Contract Data:
- Filename: ${workflow.contract_data?.filename}
- Size: ${workflow.contract_data?.size} bytes
- Uploaded: ${new Date(workflow.contract_data?.uploaded_at).toLocaleString()}
`;

            if (workflow.analysis_results && workflow.current_step !== 'error') {
                details += `
Analysis Results:
- Risk Level: ${workflow.analysis_results.risk_assessment?.risk_level}
- Risk Score: ${(workflow.analysis_results.risk_assessment?.overall_risk * 100).toFixed(1)}%
- Clauses Found: ${workflow.analysis_results.clauses?.length}
- Word Count: ${workflow.analysis_results.word_count}
`;
            }

            if (workflow.approval_decision) {
                details += `
Approval Decision:
- Decision: ${workflow.approval_decision.decision}
- By: ${workflow.approval_decision.approved_by}
- Reason: ${workflow.approval_decision.reason}
- Signature Hash: ${workflow.approval_decision.signature_hash || 'N/A'}
- Timestamp: ${new Date(workflow.approval_decision.timestamp).toLocaleString()}
`;
            }

            if (workflow.messages?.length > 0) {
                details += `
Agent Messages:
`;
                workflow.messages.forEach((msg, i) => {
                    details += `${i + 1}. [${msg.type}] ${msg.content}\n`;
                });
            }

            alert(details);
        } catch (error) {
            alert('Error loading workflow details: ' + error.message);
        }
    }

    // Update system status
    async function updateSystemStatus() {
        try {
            const response = await fetch('/health');
            const status = await response.json();
            document.getElementById('system-status').innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div style="padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 5px solid #007bff;">
                        <strong>Orchestrator Status</strong><br>
                        ${status.status.toUpperCase()}
                    </div>
                    <div style="padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 5px solid #007bff;">
                        <strong>Active Workflows</strong><br>
                        ${status.active_workflows}
                    </div>
                    <div style="padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 5px solid #007bff;">
                        <strong>LangGraph Engine</strong><br>
                        Active
                    </div>
                    <div style="padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 5px solid #007bff;">
                        <strong>MCP Server</strong><br>
                        Active (Port 8900)
                    </div>
                </div>
                <p style="margin-top: 10px;">Last checked: ${new Date(status.timestamp).toLocaleTimeString()}</p>
            `;
        } catch (error) {
            document.getElementById('system-status').innerHTML = `<p class="status error">Error connecting to system status: ${error.message}</p>`;
        }
    }

    // Initial setup on load
    document.addEventListener('DOMContentLoaded', updateUI);
    // Periodically refresh workflows
    setInterval(() => {
        if(userToken) {
            refreshWorkflows();
        }
    }, 15000);
    setInterval(updateSystemStatus, 5000);

</script>

</body>
</html>
"""

async def start_mcp_server():
    """Start the MCP server"""
    mcp_server = MCPServer(CONFIG["mcp_port"])
    config = uvicorn.Config(
        mcp_server.app, 
        host="0.0.0.0", 
        port=CONFIG["mcp_port"], 
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def start_orchestrator():
    """Start the main orchestrator"""
    orchestrator = ContractApprovalOrchestrator(CONFIG["orchestrator_port"])
    config = uvicorn.Config(
        orchestrator.app, 
        host="0.0.0.0", 
        port=CONFIG["orchestrator_port"], 
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Main application entry point"""
    
    # Check command-line arguments for different startup modes
    startup_mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if startup_mode == "full":
        # Start both MCP server and orchestrator (FastAPI)
        await asyncio.gather(
            start_mcp_server(),
            start_orchestrator()
        )
    elif startup_mode == "mcp":
        # Start only the MCP server
        await start_mcp_server()
    elif startup_mode == "orchestrator":
        # Start only the orchestrator (assumes MCP is running elsewhere)
        await start_orchestrator()
    elif startup_mode == "streamlit":
        # Start the Streamlit interface (placeholder)
        print("Streamlit mode selected, interface code is omitted for terminal environment.")
        pass
    elif startup_mode == "gradio":
        # Start the Gradio interface (placeholder)
        print("Gradio mode selected, interface code is omitted for terminal environment.")
        pass
    else:
        print(f"Unknown startup mode: {startup_mode}")

if __name__ == "__main__":
    # Handle different startup modes
    import asyncio
    import sys
    
    # Check if necessary packages are available (a helpful runtime check)
    try:
        import PyPDF2
    except ImportError:
        print("FATAL ERROR: PyPDF2 is not installed. Please run 'pip install PyPDF2'.")
        sys.exit(1)
    
    print("="*60)
    print("🤖 CONTRACT APPROVAL SYSTEM MVP")
    print("🔧 LangGraph Agents + MCP + Orchestration")
    print("="*60)
    print()
    
    # Usage instructions
    print("🚀 Usage modes:")
    print("   python complete_mvp_langgraph(1).py              # Full system (FastAPI)")
    print("   python complete_mvp_langgraph(1).py streamlit    # Streamlit interface (requires streamlit)")
    print("   python complete_mvp_langgraph(1).py gradio       # Gradio interface (requires gradio)")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down Contract Approval System...")
    except Exception as e:
        print(f"\n❌ Uncaught Fatal Error: {e}")